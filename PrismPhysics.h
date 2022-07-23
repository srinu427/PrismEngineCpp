#pragma once
#include "CollisionStructs.h"
#include "SimpleThreadPooler.h"
#include <mutex>

using namespace collutils;
class PrismPhysics
{
public:
	std::vector<PolyCollMesh*>* dmeshes;
	std::vector<PolyCollMesh*>* dmeshes_future;
	std::vector<PolyCollMesh*>* lmeshes;
	std::vector<PolyCollMesh*>* lmeshes_future;

	std::vector<std::vector<CollCache>> dd_ccache;
	std::vector<std::vector<CollCache>> dl_ccache;

	std::mutex new_mesh_lock;

	SimpleThreadPooler* thread_pool;

	PrismPhysics();
	~PrismPhysics();

	void add_pcmesh(PolyCollMesh* pcmesh, bool dynm=false);
	//cube
	void gen_and_add_pcmesh(glm::vec3 ccenter, glm::vec3 uax, glm::vec3 vax, float ulen, float vlen, float tlen, float face_thickness, float face_friction, bool dynm=false);
	//plane
	void gen_and_add_pcmesh(glm::vec3 pcenter, glm::vec3 uax, glm::vec3 vax, float ulen, float vlen, float face_thickness, float face_friction, bool dynm = false);
	//cylinder
	void gen_and_add_pcmesh(std::vector<glm::vec3> top_face_points, glm::vec3 cylinder_depth_dir, float cylinder_depth, float face_thickness, float face_friction, bool dynm = false);
	//plane
	void gen_and_add_pcmesh(std::vector<glm::vec3> plane_points, float face_thickness, float face_friction, bool dynm = false);
	CollCache get_sep_plane(PolyCollMesh* pm1, PolyCollMesh* pm2);
	void validate_spl_dl(int did, int lid, bool* res);
	void run_physics(int rt_ms);
	DynamicCollPoint find_ray_first_coll(glm::vec3 raystart, glm::vec3 raydir);

private:
	static void advance_mesh_one_step(PolyCollMesh* pm);
	void run_physics_one_step();
};

