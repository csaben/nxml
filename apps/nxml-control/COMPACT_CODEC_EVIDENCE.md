# Compact edge codec evidence

Artifact inspection on 2026-07-19 used deployed commit
`1b3a7670d2635ced8ee9b12acbbb60a3ec9da557` and the authenticated,
catalog-resolved `nxml.segment-artifact-inspection.v1` endpoint. Every outer
object and exact video/actions/events member matched its catalog size and
SHA-256 before probing or decoding.

All six videos have the same observed profile: Matroska (`matroska,webm` from
ffprobe), H.264 Main level 3.2, YUV420P, 1280x720, `r_frame_rate=60/1`,
`avg_frame_rate=60/1`, `time_base=1/1000`, GOP 60 at every measured keyframe
interval, and zero B-frames. The strict compatibility identifier is therefore
`nxml.compact-h264-main32-720p60.v1`.

H.264 level checks are explicit: `ceil(1280/16) * ceil(720/16) = 3600`
macroblocks/frame; at 60 fps this is 216,000 macroblocks/s, exactly Level 3.2
MaxMBPS, while 3,600 is below MaxFS 5,120. Main Level 3.2 MaxBR is 20,000,000
bit/s. Every measured container bitrate below is within that limit.

| Seq | Segment SHA-256 | Outer bytes | Video / actions / events SHA-256 | Bit/s | Frames = rows |
|---:|---|---:|---|---:|---:|
| 0 | `effde146c4f68ac9774e17c150e866d75d6071b71e104f01561f047363683f94` | 60,416,000 | `e1c6c5f5fb12716334242b4f923525ff39955218615ed44dc1127174cc961094` / `325f533bb5750838b4e5c4fdac5c77aef19a0ea5759c0aafb6459d6399e763c6` / `488aa5ceff62da188e687f5b4fd20ea50bdb93e964788b9f824db513871ee0a7` | 16,070,878 | 1,801 |
| 1 | `f44b33c9adf8068c43c6226f245beeeccc3f827bd66b86ce2fe5084692c468da` | 60,416,000 | `36d89616fe981e71bffe4efe860184d4c3d2332b6366c8ff69f7ef246a5fcf14` / `de0dbdb9b83916920eb7d6e02e5a0cb11678132a7c9ea1f09be97a7b7e29b71d` / `488aa5ceff62da188e687f5b4fd20ea50bdb93e964788b9f824db513871ee0a7` | 16,072,277 | 1,801 |
| 2 | `d3e67d229dbe2c3ee48c3a6fdf88cc6545564cd1623c05defb3e79fed4809fcc` | 60,416,000 | `80f9943b95cc273351283b16c94f27c9aaf8033da73907ee96168f8ee60a1ad3` / `4f86aa9cc943106f7a6c5d97ddac032e7a602183109bd8826de7ea51e0d6b8eb` / `488aa5ceff62da188e687f5b4fd20ea50bdb93e964788b9f824db513871ee0a7` | 16,072,123 | 1,801 |
| 3 | `9b5d80a46e6b08ae57fba2ed8ee05bca4c46a7751e809deec665a0877fa50464` | 31,897,600 | `2342363b7ae84d2eca4e0e66db9a813e6189d0599afb4986697e6bbd63266b63` / `b6d1d3a06faa471af8ac90d40697f74c40c3acadf0cd4328cb98a1bcdc9965a2` / `488aa5ceff62da188e687f5b4fd20ea50bdb93e964788b9f824db513871ee0a7` | 16,098,271 | 949 |
| 0 | `05e9cae7af059f733d9a5369eaf0e1375d14f80faa6504726e26dd1e43808082` | 60,416,000 | `3a643b1ddb434a5cf099b09abb2f14978cac792bbac0d468d1a73a0c78659d3c` / `55c9acfd705ce22e9034a800a1afac645a63b332eced32bc3ed648f64fe934e9` / `488aa5ceff62da188e687f5b4fd20ea50bdb93e964788b9f824db513871ee0a7` | 16,071,854 | 1,801 |
| 1 | `ea7f44b2a63544174241039609daf4b3af2aa1b223f574a8d3f23b94278d9727` | 30,402,560 | `6b5b20c91faf427684ce26a45d082284ce32ea2787c3c07502af8e34349b0b4a` / `92cde0090f47302f7e742f0d8f97cbe2c03029bfda2c02312c40b94d9ec28713` / `488aa5ceff62da188e687f5b4fd20ea50bdb93e964788b9f824db513871ee0a7` | 16,125,611 | 903 |

Episode `8f9081d8-e881-42ae-a0f6-78bf9bc92f0b` has close
`sha256:88f59ef16357f0b48e75d40591f648298a48ca5cf84a0ab70c2b6ef69b10e2e0`;
its ordered half-open boundaries are contiguous from `33976779138584` through
`34082602550734`. Episode `e7384978-b106-47c7-a142-26e060385353` has close
`sha256:4b247ccc93ce4e19789fede5bf2ebabdaf36210d2b353ab091a39c9db33e659d`;
its boundaries are contiguous from `34216630653025` through `34261678695014`.

Both episodes remain explicitly `training_eligible=false` and excluded by
snapshot `sha256:a99d3050ca05b9c8a2626b3ab9298ab6b5ffbc7f12b773e633e273c3fe0c4625`.
The offline publisher dry-run refuses both episode IDs before considering codec
compatibility. No gameplay bytes were published.

The earlier immutable synthetic shard at Hub revision
`f7d828632aacd6c0d01c5d90e1bfcdd313839bbb` is High/4.2. It remains preserved
as synthetic infrastructure evidence but is superseded and incompatible with
the artifact-derived edge profile identifier.
