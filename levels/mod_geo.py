import glm

def vec_to_str(v):
    return str(v.x) + ' ' + str(v.y) + ' ' + str(v.z)

def anim_data_to_str(adata):
    out = []
    for aname, sadata in adata.items():
        aout = []
        aout += ["LANI"]
        aout += [str(len(sadata["steps"]))]
        for astep in sadata["steps"]:
            aout += [vec_to_str(sadata["steps"]["duration"])]
            aout += [vec_to_str(sadata["steps"]["init_loc"])]
            aout += [vec_to_str(sadata["steps"]["final_loc"])]
        aout += ["LOOP" if sadata["loop"] else "ONCE"]
        aout += [aname + '\n']
        out += ["  ".join(aname)]
    return out
        

class GeoModder:
    def __init__(self):
        self.geo_data = {}

    def add_plane_uv(self, name, epsilon, friction, center, u, v, ulen, vlen):
        self.geo_data[name] = {}
        self.geo_data[name]["type"] = "plane_uv"
        self.geo_data[name]["epsilon"] = epsilon
        self.geo_data[name]["friction"] = friction
        self.geo_data[name]["center"] = center
        self.geo_data[name]["u"] = u
        self.geo_data[name]["v"] = v
        self.geo_data[name]["ulen"] = ulen
        self.geo_data[name]["vlen"] = vlen


    def make_in_cube(self, name, center, u, v, ulen, vlen, tlen):
        u = glm.normalize(u)
        v = glm.normalize(v)
        t = glm.cross(u, v)
        self.add_plane_uv(name + "_0", center + u * (ulen/2.0), t, v, tlen, vlen)
        self.add_plane_uv(name + "_1", center - u * (ulen/2.0), v, t, vlen, tlen)
        self.add_plane_uv(name + "_2", center + v * (vlen/2.0), u, t, ulen, tlen)
        self.add_plane_uv(name + "_3", center - v * (vlen/2.0), t, u, tlen, ulen)
        self.add_plane_uv(name + "_4", center + t * (tlen/2.0), v, u, vlen, ulen)
        self.add_plane_uv(name + "_5", center - t * (tlen/2.0), u, v, ulen, vlen)

    def write_to_file(self, fname):
        with open(fname, 'w') as fw:
            for name, gd in self.geo_data.items():
                if gd["type"] == "plane_uv":
                    linel = []
                    linel += ["PUVL"]
                    linel += [gd["epsilon"]]
                    linel += [gd["friction"]]
                    linel += [vec_to_str(gd["center"])]
                    linel += [vec_to_str(gd["u"])]
                    linel += [vec_to_str(gd["v"])]
                    linel += [gd["ulen"]]
                    linel += [gd["vlen"]]
                    linel += ['\/\/' + name + '\n']
                    fw.write("  ".join(linel))
                    if "anims" in gd:
                        anim_lines = anim_data_to_str(gd["anims"])
                        for anline in anim_lines:
                            fw.write(anline)
                    continue
                if gd["type"] == "cylinder_uv":
                    linel = []
                    linel += ["CUVL"]
                    linel += [gd["epsilon"]]
                    linel += [gd["friction"]]
                    linel += [vec_to_str(gd["center"])]
                    linel += [vec_to_str(gd["u"])]
                    linel += [vec_to_str(gd["v"])]
                    linel += [gd["ulen"]]
                    linel += [gd["vlen"]]
                    linel += [gd["tlen"]]
                    linel += ['\/\/' + name + '\n']
                    fw.write("  ".join(linel))
                    if "anims" in gd:
                        anim_lines = anim_data_to_str(gd["anims"])
                        for anline in anim_lines:
                            fw.write(anline)
                    continue
                if gd["type"] == "plane_np":
                    linel = []
                    linel += ["PNSP"]
                    linel += [gd["epsilon"]]
                    linel += [gd["friction"]]
                    linel += [str(len(gd["points"]))]
                    for pnt in gd["points"]:
                        linel += [vec_to_str(pnt)]
                    linel += ['\/\/' + name + '\n']
                    fw.write("  ".join(linel))
                    if "anims" in gd:
                        anim_lines = anim_data_to_str(gd["anims"])
                        for anline in anim_lines:
                            fw.write(anline)
                    continue
                if gd["type"] == "cylinder_np":
                    linel = []
                    linel += ["CNPH"]
                    linel += [gd["epsilon"]]
                    linel += [gd["friction"]]
                    linel += [str(len(gd["points"]))]
                    for pnt in gd["points"]:
                        linel += [vec_to_str(pnt)]
                    linel += [gd["tlen"]]
                    linel += ['\/\/' + name + '\n']
                    fw.write("  ".join(linel))
                    if "anims" in gd:
                        anim_lines = anim_data_to_str(gd["anims"])
                        for anline in anim_lines:
                            fw.write(anline)
                    continue

    def run(self):
        while True:
            inp = input("input:\n")
            inp_list = inp.split();



def main():
    gmod = GeoModder()
    gmod.run()
    


if __name__ =="__main__":
    main()