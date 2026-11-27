//! A d16 image load through views of integer and of float formats, run on
//! the host GPU and WARP: the view's kind, from the host's word for its heap
//! slot (`SLOT_KINDS`), picks the conversion, not the value the load
//! returned (0xFFFF8000 is -32768 from an `R16_SINT` view and a NaN from an
//! `R32_FLOAT` one).

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, SLOT_KINDS, VIEW_INTEGER, VIEW_SIGNED, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// A wave32 compute program, its table of views at s0: an image at entry 0
/// (and 1), a raw UAV at entry 2 (dword 16). It loads texel (0, 0) of the
/// image d16 and stores the result's dword at 0.
const D16_LOAD: &[u32] = &[
    0xBE82_0300, // s_mov_b32 s2, s0
    0xBE83_0380, // s_mov_b32 s3, 0
    0xF40C_0201,
    0xFA00_0000, // s_load_dwordx8 s[8:15], s[2:3], 0
    0xF408_0501,
    0xFA00_0040, // s_load_dwordx4 s[20:23], s[2:3], 0x40
    0x7E0C_0280, // v_mov_b32 v6, 0
    0x7E0E_0280, // v_mov_b32 v7, 0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xF000_1108,
    0x8002_0D06, // image_load v13, v[6:7], s[8:15] dmask:0x1 dim:2D unorm d16
    0xBF8C_0000, // s_waitcnt 0
    0xE070_0000,
    0x8005_0D00, // buffer_store_dword v13, off, s[20:23], 0
    0xBF81_0000, // s_endpgm
];

fn build(words: &[u32]) -> HostShader {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 8,
                user_data: 0,
                slot: 0,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 16,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 0,
                slot: 0,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: Vec::new(),
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    shader
}

/// Whether `h` is a half-precision NaN.
fn half_nan(h: u32) -> bool {
    h & 0x7C00 == 0x7C00 && h & 0x03FF != 0
}

/// (Scarlett, `VICTORY_D16_BITS`) The same texel bits through a view the
/// host marks as of an integer format and through one it does not: the
/// first keeps the low 16 bits, the second converts a float to a half
/// (=0: both convert). Under the earlier rule, which guessed from the
/// value, both views gave 0x8000 and 0x1234.
#[test]
fn a_d16_image_load_converts_by_the_views_kind_not_its_value() {
    let shader = build(D16_LOAD);
    let on = std::env::var_os("VICTORY_D16_BITS").is_none_or(|v| v != "0");
    assert_eq!(shader.root_srvs.contains(&SLOT_KINDS), on, "the table of kinds: {:?}", shader.root_srvs);
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        // The table starts at heap slot 0; the user data's other words, none.
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let load = |bits: u32, kind: u32| {
            let texel = [[f32::from_bits(bits), 0.0, 0.0, 0.0]];
            let mut kinds = vec![0u32; 16];
            kinds[0] = kind;
            let mut roots = vec![(0, Root::Constants(ud.clone()))];
            if on {
                roots.push((1, Root::View { srv: true, data: kinds }));
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots,
                        views: &[View::Texture(1, 1, &texel), View::Texture(1, 1, &texel), View::RawUav(4)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            out[0][0]
        };
        let signed = VIEW_INTEGER | VIEW_SIGNED;
        // -32768 sign-extended from an R16_SINT view, a NaN from a float one.
        let (int, float) = (load(0xFFFF_8000, signed), load(0xFFFF_8000, 0));
        assert!(half_nan(float), "{label}: a float view's NaN gave {float:#x}, not a half NaN");
        if on {
            assert_eq!(int, 0x8000, "{label}: the integer view's bits were converted as a float");
        } else {
            assert!(half_nan(int), "{label}: VICTORY_D16_BITS=0 kept the bits ({int:#x})");
        }
        // 0x1234 from an R16_UINT view, a denormal float (a half's 0) from a float one.
        let (int, float) = (load(0x1234, VIEW_INTEGER), load(0x1234, 0));
        assert_eq!(float, 0, "{label}: a float view's denormal was taken as integer bits");
        assert_eq!(int, if on { 0x1234 } else { 0 }, "{label}: the integer view's 16 bits");
        ran += 1;
    }
    assert!(ran > 0, "no SM 6.6 device available");
}
