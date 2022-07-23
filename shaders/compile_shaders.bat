glslc light_smap.vert -o light_smap.vert.spv
glslc light_smap.frag -o light_smap.frag.spv

glslc gbuffer.vert -o gbuffer.vert.spv
glslc gbuffer.frag -o gbuffer.frag.spv

glslc screensize.vert -o screensize.vert.spv
glslc ambient.frag -o ambient.frag.spv
glslc finalimage.frag -o finalimage.frag.spv