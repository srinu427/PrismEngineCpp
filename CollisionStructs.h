#pragma once
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <vector>
#include <queue>
#include "vkstructs.h"
#include "SimpleThreadPooler.h"
#include "ModelStructs.h"

void print_vec(glm::vec3 v);

void print_vec(glm::vec4 v);

namespace collutils {

	struct CirclePlane
	{
		glm::vec4 equation;
		glm::vec3 n;
		glm::vec3 center;
		float height, radius;

		CirclePlane(glm::vec3 planeNormal, glm::vec3 planeCenter, float circleRadius, float thickness = 0);
		int point_status(glm::vec3 p);
	};

	struct Sphere
	{
		glm::vec3 center;
		float radius;

		Sphere(glm::vec3 sphereCenter, float sphereRadius);
		int point_status(glm::vec3 p);
	};

	struct PlaneMeta {
		std::vector<size_t> vinds;
		size_t vinds_size = 0;
		glm::vec3 normal = glm::vec3(0);
		glm::vec4 equation = glm::vec4(0);

		void process_plane(std::vector<glm::vec3>* fverts);
	};

	struct DynamicCollPoint {
		bool will_collide = false;
		glm::vec3 relative_disp = glm::vec3(0);
		size_t obj_index = 0;
		float time = 0;
	};

	struct CollPoint {
		bool will_collide = false;
		glm::vec3 displacement1 = glm::vec3(0);
		glm::vec3 displacement2 = glm::vec3(0);
		float time = 0;
	};

	struct MeshCollData {
		bool will_collide = false;
		float time = 0;
		bool second_owns_bplane = false;
		glm::vec3 bound_dir = glm::vec3(0);
		glm::vec4 bound_plane = glm::vec4(0);
		glm::vec3 displacement1 = glm::vec3(0);
		glm::vec3 displacement2 = glm::vec3(0);

		uint32_t flags = 0;
	};

	struct PolyCollMesh {
		std::vector<glm::vec3> verts;
		size_t verts_size = 0;
		std::vector<PlaneMeta> faces;
		size_t faces_size = 0;
		std::vector<glm::ivec2> edges;
		size_t edges_size = 0;

		float face_epsilon = 0.1f;
		float friction = 1;
		bool inf_mass = true;
		float mass = 1;

		std::unordered_map<std::string, BoneAnimData> anims;
		std::vector<std::string> running_anims;

		glm::mat3 _uvw = glm::mat3(1);

		glm::vec3 _init_center = glm::vec3(0);
		glm::vec3 _center = glm::vec3(0);
		glm::vec3 _vel = glm::vec3(0);
		glm::vec3 _acc = glm::vec3(0);
		glm::vec3 _bvel = glm::vec3(0);
		glm::vec3 _bacc = glm::vec3(0);

		bool controlled = false;

		std::string coll_behav = "physics";
		std::vector<std::string> coll_behav_args;

		bool in_renderer = false;
		size_t renderer_id = 0;
		size_t logicmgr_id = 0;

		PolyCollMesh();
		PolyCollMesh(PolyCollMesh* pcm);
		~PolyCollMesh();

		Prism::Mesh gen_mesh();
		void apply_displacement(glm::vec3 disp);
		void apply_LRS(LRS mlrs);
		void updateAnim(std::string animName, int gap_ms);
		int getAnimEventTime(std::string animName);
		bool is_point_on_face_bounds(int plane_idx, glm::vec3 p);
		bool is_point_on_face_bounds(int plane_idx, glm::vec3 p, float after_time);
		CollPoint find_ray_first_coll(glm::vec3 raystart, glm::vec3 raydir);
		CollPoint check_point_coll_with_face(int plane_idx, glm::vec3 p, glm::vec3 pvel, glm::vec3 pacc, float sim_time);
		CollPoint check_point_list_coll_with_face(int plane_idx, glm::vec3* p_array, int point_count, glm::vec3 pvel, glm::vec3 pacc, float sim_time);
	};

	struct TDCollisionMeta
	{
		float m1 = 1;
		float m2 = 1;
		glm::vec3 vel1;
		glm::vec3 vel2;
		glm::vec3 collision_line;
	};

	struct DynBound {
		glm::vec4 _plane;
		glm::vec3 _dir;
		glm::vec3 _vel;
		glm::vec3 _acc;
		float friction = 0;
		float epsilon = 0;
	};

	struct CollCache {
		glm::vec4 sep_plane = glm::vec4(0);
		glm::vec3 _dir = glm::vec3(0);
		bool m1side = false;
		bool m2side = false;
		size_t sep_plane_idx = 0;
	};

	TDCollisionMeta make_basic_collision(TDCollisionMeta cmeta, float eratio = 1.0f);

	glm::vec3 project_vec_on_plane(glm::vec3 vToProj, glm::vec3 planeN);
	glm::vec3 normalize_vec_in_dir(glm::vec3 vToProj, glm::vec3 planeN, float norm_value = 0);

}
