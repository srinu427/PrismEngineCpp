#include "CollisionStructs.h"
#include <set>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <glm/gtc/type_ptr.hpp>


void print_vec(glm::vec3 v) {
	std::cout << v.x << ',' << v.y << ',' << v.z << std::endl;
}

void print_vec(glm::vec4 v) {
	std::cout << v.x << ',' << v.y << ',' << v.z << ',' << v.w << std::endl;
}

collutils::CirclePlane::CirclePlane(glm::vec3 planeNormal, glm::vec3 planeCenter, float circleRadius, float thickness)
{
	n = glm::normalize(planeNormal);
	height = abs(thickness);
	radius = circleRadius;
	center = planeCenter;
	equation = glm::vec4(n, -glm::dot(n, planeCenter));
}

int collutils::CirclePlane::point_status(glm::vec3 p)
{
	float ndist = glm::dot(glm::vec4(p, 1), equation);
	glm::vec3 q = p - ndist * n;
	if (glm::length(q - center) <= radius) return (0 <= ndist && ndist <= height) ? 1 : 2;
	else return 0;
}

collutils::Sphere::Sphere(glm::vec3 sphereCenter, float sphereRadius)
{
	center = sphereCenter;
	radius = sphereRadius;
}

int collutils::Sphere::point_status(glm::vec3 p)
{
	return (glm::length(p - center) <= radius) ? 1 : 0;
}

void collutils::PlaneMeta::process_plane(std::vector<glm::vec3>* fverts) {
	normal = glm::normalize(glm::cross((*fverts)[vinds[1]] - (*fverts)[vinds[0]], (*fverts)[vinds[2]] - (*fverts)[vinds[1]]));
	equation = glm::vec4(normal, -glm::dot(normal, (*fverts)[vinds[0]]));
}

collutils::PolyCollMesh::PolyCollMesh()
{
}

collutils::PolyCollMesh::PolyCollMesh(collutils::PolyCollMesh* pcm)
{
	verts = pcm->verts;
	verts_size = pcm->verts_size;
	faces = pcm->faces;
	faces_size = pcm->faces_size;
	edges = pcm->edges;
	edges_size = pcm->edges_size;

	mass = pcm->mass;
	inf_mass = pcm->inf_mass;
	friction = pcm->friction;
	face_epsilon = pcm->face_epsilon;

	anims = pcm->anims;
	running_anims = pcm->running_anims;

	_uvw = pcm->_uvw;
	_center = pcm->_center;
	_init_center = pcm->_init_center;
	_vel = pcm->_vel;
	_bvel = pcm->_bvel;
	_acc = pcm->_acc;
	_bacc = pcm->_bacc;
}

collutils::PolyCollMesh::~PolyCollMesh()
{
	verts.clear();
	edges.clear();
	for (uint32_t i = 0; i < faces_size; i++) {
		faces[i].vinds.clear();
	}
	faces.clear();
}

Prism::Mesh collutils::PolyCollMesh::gen_mesh()
{
	std::vector<Prism::Vertex> vlist;
	for (uint32_t i = 0; i < faces_size; i ++ ){
		PlaneMeta* plmet = &faces[i];
		glm::vec3 uaxn = glm::normalize(verts[plmet->vinds[1]] - verts[plmet->vinds[0]]);
		glm::vec3 vaxn = glm::cross(plmet->normal, uaxn);

		for (uint32_t ti = 0; ti < plmet->vinds_size - 2; ti++) {
			Prism::Vertex vtmp;
			vtmp.pos = verts[plmet->vinds[0]];
			vtmp.normal = plmet->normal;
			vtmp.texCoord = glm::vec2(0, 0);

			vlist.push_back(vtmp);

			vtmp.pos = verts[plmet->vinds[ti + 1]];
			vtmp.normal = plmet->normal;
			vtmp.texCoord = glm::vec2(0.1 * glm::dot(uaxn, verts[plmet->vinds[ti + 1]] - verts[plmet->vinds[0]]), 0.1 * glm::dot(vaxn, verts[plmet->vinds[ti + 1]] - verts[plmet->vinds[0]]));

			vlist.push_back(vtmp);

			vtmp.pos = verts[plmet->vinds[ti + 2]];
			vtmp.normal = plmet->normal;
			vtmp.texCoord = glm::vec2(0.1 * glm::dot(uaxn, verts[plmet->vinds[ti + 2]] - verts[plmet->vinds[0]]), 0.1 * glm::dot(vaxn, verts[plmet->vinds[ti + 2]] - verts[plmet->vinds[0]]));

			vlist.push_back(vtmp);
		}
	}

	Prism::Mesh mout;
	mout.add_vertices(vlist);

	return mout;
}

void collutils::PolyCollMesh::apply_displacement(glm::vec3 disp)
{
	_center += disp;
	for (uint32_t i = 0; i < verts_size; i++) {
		verts[i] += disp;
	}
	for (uint32_t i = 0; i < faces_size; i++) {
		faces[i].process_plane(&verts);
	}
}

void collutils::PolyCollMesh::apply_LRS(LRS mlrs)
{
	glm::mat4 tmat = mlrs.getTMatrix();
	glm::vec3 cdisp = _init_center - _center;
	apply_displacement(cdisp);
	for (uint32_t i = 0; i < verts_size; i++) {
		verts[i] = glm::vec3(mlrs.rotate * glm::vec4(verts[i], 1));
	}
	cdisp = mlrs.location - _center;
	apply_displacement(cdisp);
}

void collutils::PolyCollMesh::updateAnim(std::string animName, int gap_ms)
{
	LRS animLRS = anims[animName].transformAfterGap(gap_ms);
	glm::vec3 disp = animLRS.location - _center;
	apply_displacement(disp);
	_vel = anims[animName].getNextVelHint();
	_bvel = anims[animName].getNextVelHint();
	if (animLRS.anim_finished) {
		running_anims.clear();
		_vel = glm::vec3(0);
		_bvel = glm::vec3(0);
	}
}

int collutils::PolyCollMesh::getAnimEventTime(std::string animName)
{
	return anims[animName].getNextEventTime();
}

bool collutils::PolyCollMesh::is_point_on_face_bounds(int plane_idx, glm::vec3 p)
{
	glm::vec4 pv4 = glm::vec4(p, 1);
	PlaneMeta* plmet = &faces[plane_idx];
	for (uint32_t i = 0; i < plmet->vinds_size; i++) {
		glm::vec3 tmp = glm::normalize(glm::cross(plmet->normal, verts[plmet->vinds[(i + 1) % plmet->vinds_size]] - verts[plmet->vinds[i]]));
		glm::vec4 bplanet = glm::vec4(tmp, -glm::dot(tmp, verts[plmet->vinds[i]]));
		if (glm::dot(bplanet, pv4) < 0) return false;
	}
	return true;
}

bool collutils::PolyCollMesh::is_point_on_face_bounds(int plane_idx, glm::vec3 p, float after_time)
{
	glm::vec4 pv4 = glm::vec4(p, 1);
	PlaneMeta* plmet = &faces[plane_idx];
	for (uint32_t i = 0; i < plmet->vinds_size; i++) {
		glm::vec3 tmp = glm::normalize(glm::cross(plmet->normal, verts[plmet->vinds[(i + 1) % plmet->vinds_size]] - verts[plmet->vinds[i]]));
		glm::vec4 bplanet = glm::vec4(tmp, -glm::dot(tmp, verts[plmet->vinds[i]] + (_vel * after_time) + (0.5f * _acc * after_time * after_time)));
		if (glm::dot(bplanet, pv4) < 0) return false;
	}
	return true;
}

collutils::CollPoint collutils::PolyCollMesh::find_ray_first_coll(glm::vec3 raystart, glm::vec3 raydir)
{
	CollPoint cpoint;
	float min_rd = 100000;
	for (uint32_t i = 0; i < faces_size; i++) {
		if (glm::dot(faces[i].normal, raydir) < 0 && abs(glm::dot(faces[i].normal, raydir)) > 0) {
			float t = -glm::dot(faces[i].equation, glm::vec4(raystart, 1))/glm::dot(faces[i].normal, raydir);
			glm::vec3 rtp = raystart + (raydir * t);
			if (t > 0 && t < min_rd && is_point_on_face_bounds(i, rtp)) {
				min_rd = t;
				cpoint.time = t;
				cpoint.will_collide = true;
				cpoint.displacement1 = rtp;
			}
		}
	}
	return cpoint;

}

collutils::TDCollisionMeta collutils::make_basic_collision(TDCollisionMeta cmeta, float eratio)
{
	float u1 = glm::dot(cmeta.vel1, cmeta.collision_line);
	float u2 = glm::dot(cmeta.vel2, cmeta.collision_line);

	float v1 = u1;
	float v2 = u2;

	float Ei = 0.5f * (cmeta.m1 * u1 * u1 + cmeta.m2 * u2 * u2);
	float P = (cmeta.m1 * u1 + cmeta.m2 * u2);

	float min_eratio = (P * P) / (Ei * (cmeta.m1 + cmeta.m2));

	TDCollisionMeta outdata = cmeta;
	
	if (min_eratio >= eratio) {
		v1 = P / (cmeta.m1 + cmeta.m2);
		outdata.vel1 = cmeta.vel1 + ((v1 - u1) * cmeta.collision_line);
		outdata.vel2 = outdata.vel1;
	}
	else {
		v2 = (P * cmeta.m2 + sqrt(Ei * eratio * (cmeta.m1 + cmeta.m2) - P * P)) / (cmeta.m2 * (cmeta.m1 + cmeta.m2));
		v1 = (P - cmeta.m2 * v2) / cmeta.m1;
		outdata.vel1 = cmeta.vel1 + ((v1 - u1) * cmeta.collision_line);
		outdata.vel2 = cmeta.vel2 + ((v2 - u2) * cmeta.collision_line);
	}
	return outdata;
}

glm::vec3 collutils::project_vec_on_plane(glm::vec3 vToProj, glm::vec3 planeN) {
	return normalize_vec_in_dir(vToProj, planeN, 0);
}

glm::vec3 collutils::normalize_vec_in_dir(glm::vec3 vToProj, glm::vec3 planeN, float norm_value)
{
	if (glm::length(planeN) == 0) return vToProj;
	glm::vec3 nPlaneN = glm::normalize(planeN);
	glm::vec3 tmpo =  vToProj + (norm_value - glm::dot(nPlaneN, vToProj)) * nPlaneN;
	tmpo = tmpo + (norm_value - glm::dot(nPlaneN, tmpo)) * nPlaneN;
	tmpo = tmpo + (norm_value - glm::dot(nPlaneN, tmpo)) * nPlaneN;
	return tmpo;
}
