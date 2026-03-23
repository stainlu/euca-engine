# euca-terrain

Heightmap terrain system: chunked mesh generation, quad-tree LOD, texture splatting, and brush editing.

Part of [EucaEngine](https://github.com/stainlu/euca-engine) -- an ECS-first game engine in Rust.

## Features

- `Heightmap` with configurable resolution, cell size, and max height
- Chunk-based mesh generation (`build_chunks`, `generate_chunk_mesh`)
- Distance-based LOD selection (`LodConfig`, `select_chunk_lod`, `select_all_lods`)
- `SplatMap` for up to 4 texture layers per terrain
- `TerrainComponent` with `TerrainLayer` material definitions
- Brush editing tools: `raise_terrain`, `lower_terrain`, `smooth_terrain`, `flatten_terrain`, `paint_splat`
- Physics integration: `generate_heightfield_colliders` and `height_at` queries
- Frustum culling for terrain chunks

## Usage

```rust
use euca_terrain::*;

let heightmap = Heightmap::flat(128, 128)
    .with_cell_size(1.0)
    .with_max_height(50.0);

let terrain = TerrainComponent::new(heightmap, 32);
let chunks = build_chunks(&terrain.heightmap, terrain.chunk_size);
```

## License

MIT
