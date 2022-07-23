#include "PrismPhysics.h"

void move_mesh_out_plane(PolyCollMesh* pm1, glm::vec4 plane_eq, float epsilon) {
	float move_dist = epsilon;
	for (int i = 0; i < pm1->verts_size; i++) {
		float vdist = glm::dot(plane_eq, glm::vec4(pm1->verts[i], 1));
		if (vdist < move_dist) {
			move_dist = vdist;
		}
	}
	pm1->apply_displacement((-move_dist + (epsilon * 0.9f)) * glm::normalize(glm::vec3(plane_eq)));
}

PrismPhysics::PrismPhysics()
{
	dmeshes = new std::vector<PolyCollMesh*>();
	dmeshes_future = new std::vector<PolyCollMesh*>();
	lmeshes = new std::vector<PolyCollMesh*>();
	lmeshes_future = new std::vector<PolyCollMesh*>();

	thread_pool = new SimpleThreadPooler(4);
	thread_pool->run();
}

PrismPhysics::~PrismPhysics()
{
	delete dmeshes;
	delete dmeshes_future;
	delete lmeshes;
	delete lmeshes_future;

	delete thread_pool;
}

void PrismPhysics::add_pcmesh(PolyCollMesh* pcmesh, bool dynm)
{
	new_mesh_lock.lock();
	PolyCollMesh* pmf = new PolyCollMesh(pcmesh);
	if (dynm) {
		dmeshes->push_back(pcmesh);
		dmeshes_future->push_back(pmf);
		dd_ccache.resize(dd_ccache.size() + 1);
		dd_ccache[dd_ccache.size() - 1] = std::vector<CollCache>(dmeshes->size() - 1);
		for (int i = 0; i < dmeshes->size() - 1; i++) {
			dd_ccache[dd_ccache.size() - 1][i] = get_sep_plane((*dmeshes)[dd_ccache.size() - 1], (*dmeshes)[i]);
		}
		dl_ccache.resize(dl_ccache.size() + 1);
		dl_ccache[dl_ccache.size() - 1] = std::vector<CollCache>(lmeshes->size());
		for (int i = 0; i < lmeshes->size(); i++) {
			dl_ccache[dl_ccache.size() - 1][i] = get_sep_plane((*dmeshes)[dl_ccache.size() - 1], (*lmeshes)[i]);
		}
	}
	else {
		lmeshes->push_back(pcmesh);
		lmeshes_future->push_back(pmf);
		for (int i = 0; i < dmeshes->size(); i++) {
			dl_ccache[i].resize(lmeshes->size());
			dl_ccache[i][lmeshes->size() - 1] = get_sep_plane((*dmeshes)[i], (*lmeshes)[lmeshes->size() - 1]);
		}
	}
	new_mesh_lock.unlock();
}

void PrismPhysics::gen_and_add_pcmesh(glm::vec3 ccenter, glm::vec3 uax, glm::vec3 vax, float ulen, float vlen, float tlen, float face_thickness, float face_friction, bool dynm)
{
	new_mesh_lock.lock();
	PolyCollMesh* pm1 = new PolyCollMesh();
	pm1->_center = ccenter;
	pm1->_init_center = ccenter;
	pm1->face_epsilon = face_thickness;
	pm1->friction = face_friction;

	glm::vec3 uaxn = glm::normalize(uax);
	glm::vec3 vaxn = glm::normalize(vax);
	glm::vec3 cylinder_depth_dir = -glm::cross(uaxn, vaxn);

	pm1->_uvw = glm::mat3(uaxn, vaxn, -cylinder_depth_dir);

	float cylinder_depth = tlen;

	glm::vec3 tmpu = 0.5f * ulen * uaxn;
	glm::vec3 tmpv = 0.5f * vlen * vaxn;
	glm::vec3 tmpt = 0.5f * tlen * -cylinder_depth_dir;

	std::vector<glm::vec3> top_face_points(4);

	top_face_points[0] = ccenter + tmpu + tmpv + tmpt;
	top_face_points[1] = ccenter - tmpu + tmpv + tmpt;
	top_face_points[2] = ccenter - tmpu - tmpv + tmpt;
	top_face_points[3] = ccenter + tmpu - tmpv + tmpt;

	glm::vec3 cn = glm::normalize(cylinder_depth_dir);
	glm::vec3 cno = cn * cylinder_depth;

	pm1->verts_size = 2 * top_face_points.size();
	pm1->verts.resize(pm1->verts_size);
	memcpy(pm1->verts.data(), top_face_points.data(), top_face_points.size() * sizeof(glm::vec3));

	pm1->faces_size = 2 + top_face_points.size();
	pm1->faces.resize(pm1->faces_size);

	pm1->edges_size = 3 * top_face_points.size();
	pm1->edges.resize(pm1->edges_size);

	pm1->faces[0].vinds_size = top_face_points.size();
	pm1->faces[0].vinds.resize(pm1->faces[0].vinds_size);

	pm1->faces[1].vinds_size = top_face_points.size();
	pm1->faces[1].vinds.resize(pm1->faces[1].vinds_size);

	for (size_t i = 0; i < (pm1->verts_size >> 1); i++) {
		pm1->verts[pm1->verts_size - i - 1] = pm1->verts[i] + cno;

		size_t a = (i + 1) % (pm1->verts_size >> 1);
		size_t b = pm1->verts_size - 1 - ((i + 1) % (pm1->verts_size >> 1));

		pm1->faces[0].vinds[i] = i;
		pm1->faces[1].vinds[i] = (pm1->verts_size >> 1) + i;

		PlaneMeta* plmet = &pm1->faces[i + 2];
		plmet->vinds_size = 4;
		plmet->vinds.resize(4);
		plmet->vinds[0] = i;
		plmet->vinds[1] = pm1->verts_size - i - 1;
		plmet->vinds[2] = b;
		plmet->vinds[3] = a;

		pm1->edges[3 * i] = glm::ivec2(i, a);
		pm1->edges[3 * i + 1] = glm::ivec2(b, pm1->verts_size - 1 - i);
		pm1->edges[3 * i + 2] = glm::ivec2(i, pm1->verts_size - i - 1);
	}

	for (size_t i = 0; i < pm1->faces_size; i++) {
		pm1->faces[i].process_plane(&pm1->verts);
	}

	PolyCollMesh* pm2 = new PolyCollMesh(pm1);

	if (dynm) {
		dmeshes->push_back(pm1);
		dmeshes_future->push_back(pm2);
		dd_ccache.resize(dd_ccache.size() + 1);
		dd_ccache[dd_ccache.size() - 1] = std::vector<CollCache>(dmeshes->size() - 1);
		for (int i = 0; i < dmeshes->size() - 1; i++) {
			dd_ccache[dd_ccache.size() - 1][i] = get_sep_plane((*dmeshes)[dd_ccache.size() - 1], (*dmeshes)[i]);
		}
		dl_ccache.resize(dl_ccache.size() + 1);
		dl_ccache[dl_ccache.size() - 1] = std::vector<CollCache>(lmeshes->size());
		for (int i = 0; i < lmeshes->size(); i++) {
			dl_ccache[dl_ccache.size() - 1][i] = get_sep_plane((*lmeshes)[i], (*dmeshes)[dl_ccache.size() - 1]);
		}
	}
	else {
		lmeshes->push_back(pm1);
		lmeshes_future->push_back(pm2);
		for (int i = 0; i < dmeshes->size(); i++) {
			dl_ccache[i].resize(lmeshes->size());
			dl_ccache[i][lmeshes->size() - 1] = get_sep_plane((*lmeshes)[lmeshes->size() - 1], (*dmeshes)[i]);
		}
	}
	new_mesh_lock.unlock();
}

void PrismPhysics::gen_and_add_pcmesh(glm::vec3 pcenter, glm::vec3 uax, glm::vec3 vax, float ulen, float vlen, float face_thickness, float face_friction, bool dynm)
{
	new_mesh_lock.lock();
	PolyCollMesh* pm1 = new PolyCollMesh();

	pm1->face_epsilon = face_thickness;
	pm1->friction = face_friction;

	glm::vec3 uaxn = glm::normalize(uax);
	glm::vec3 vaxn = glm::normalize(vax);

	pm1->_uvw = glm::mat3(uaxn, vaxn, glm::cross(uaxn, vaxn));

	glm::vec3 tmpu = 0.5f * ulen * uaxn;
	glm::vec3 tmpv = 0.5f * vlen * vaxn;

	std::vector<glm::vec3> plane_points(4);

	plane_points[0] = pcenter + tmpu + tmpv;
	plane_points[1] = pcenter - tmpu + tmpv;
	plane_points[2] = pcenter - tmpu - tmpv;
	plane_points[3] = pcenter + tmpu - tmpv;

	pm1->verts_size = plane_points.size();
	pm1->verts.resize(pm1->verts_size);
	memcpy(pm1->verts.data(), plane_points.data(), pm1->verts_size * sizeof(glm::vec3));

	pm1->faces_size = 1;
	pm1->faces.resize(pm1->faces_size);
	pm1->faces[0].vinds_size = pm1->verts_size;
	pm1->faces[0].vinds.resize(pm1->faces[0].vinds_size);

	pm1->edges_size = pm1->verts_size;
	pm1->edges.resize(pm1->edges_size);

	for (int i = 0; i < (pm1->verts_size); i++) {
		pm1->faces[0].vinds[i] = i;
		pm1->edges[i] = glm::ivec2(i, (i + 1) % pm1->edges_size);
	}

	pm1->faces[0].process_plane(&pm1->verts);

	for (int i = 0; i < pm1->verts_size; i++) {
		pm1->_center += pm1->verts[i];
	}
	pm1->_center /= float(pm1->verts_size);
	pm1->_init_center = pm1->_center;

	PolyCollMesh* pm2 = new PolyCollMesh(pm1);

	if (dynm) {
		dmeshes->push_back(pm1);
		dmeshes_future->push_back(pm2);
		dd_ccache.resize(dd_ccache.size() + 1);
		dd_ccache[dd_ccache.size() - 1] = std::vector<CollCache>(dmeshes->size() - 1);
		for (int i = 0; i < dmeshes->size() - 1; i++) {
			dd_ccache[dd_ccache.size() - 1][i] = get_sep_plane((*dmeshes)[dd_ccache.size() - 1], (*dmeshes)[i]);
		}
		dl_ccache.resize(dl_ccache.size() + 1);
		dl_ccache[dl_ccache.size() - 1] = std::vector<CollCache>(lmeshes->size());
		for (int i = 0; i < lmeshes->size(); i++) {
			dl_ccache[dl_ccache.size() - 1][i] = get_sep_plane((*lmeshes)[i], (*dmeshes)[dl_ccache.size() - 1]);
		}
	}
	else {
		lmeshes->push_back(pm1);
		lmeshes_future->push_back(pm2);
		for (int i = 0; i < dmeshes->size(); i++) {
			dl_ccache[i].resize(lmeshes->size());
			dl_ccache[i][lmeshes->size() - 1] = get_sep_plane((*lmeshes)[lmeshes->size() - 1], (*dmeshes)[i]);
		}
	}
	new_mesh_lock.unlock();
}

void PrismPhysics::gen_and_add_pcmesh(std::vector<glm::vec3> top_face_points, glm::vec3 cylinder_depth_dir, float cylinder_depth, float face_thickness, float face_friction, bool dynm)
{
	new_mesh_lock.lock();
	PolyCollMesh* pm1 = new PolyCollMesh();

	pm1->face_epsilon = face_thickness;
	pm1->friction = face_friction;

	glm::vec3 cn = glm::normalize(cylinder_depth_dir);
	glm::vec3 cno = cn * cylinder_depth;

	pm1->verts_size = 2 * top_face_points.size();
	pm1->verts.resize(pm1->verts_size);
	memcpy(pm1->verts.data(), top_face_points.data(), top_face_points.size() * sizeof(glm::vec3));

	pm1->faces_size = 2 + top_face_points.size();
	pm1->faces.resize(pm1->faces_size);

	pm1->edges_size = 3 * top_face_points.size();
	pm1->edges.resize(pm1->edges_size);

	pm1->faces[0].vinds_size = top_face_points.size();
	pm1->faces[0].vinds.resize(pm1->faces[0].vinds_size);

	pm1->faces[1].vinds_size = top_face_points.size();
	pm1->faces[1].vinds.resize(pm1->faces[1].vinds_size);

	glm::vec3 uaxn = glm::normalize(top_face_points[1] - top_face_points[0]);
	glm::vec3 vaxn = (top_face_points[2] - top_face_points[0]);
	vaxn = glm::normalize(vaxn - glm::dot(uaxn, vaxn) * uaxn);

	pm1->_uvw = glm::mat3(uaxn, vaxn, glm::cross(uaxn, vaxn));

	for (size_t i = 0; i < (pm1->verts_size >> 1); i++) {
		pm1->verts[pm1->verts_size - i - 1] = pm1->verts[i] + cno;

		size_t a = (i + 1) % (pm1->verts_size >> 1);
		size_t b = pm1->verts_size - 1 - ((i + 1) % (pm1->verts_size >> 1));

		pm1->faces[0].vinds[i] = i;
		pm1->faces[1].vinds[i] = (pm1->verts_size >> 1) + i;

		PlaneMeta* plmet = &pm1->faces[i + 2];
		plmet->vinds_size = 4;
		plmet->vinds.resize(4);
		plmet->vinds[0] = i;
		plmet->vinds[1] = pm1->verts_size - i - 1;
		plmet->vinds[2] = b;
		plmet->vinds[3] = a;

		pm1->edges[3 * i] = glm::ivec2(i, a);
		pm1->edges[3 * i + 1] = glm::ivec2(b, pm1->verts_size - 1 - i);
		pm1->edges[3 * i + 2] = glm::ivec2(i, pm1->verts_size - i - 1);
	}

	for (int i = 0; i < pm1->faces_size; i++) {
		pm1->faces[i].process_plane(&pm1->verts);
	}
	for (int i = 0; i < pm1->verts_size; i++) {
		pm1->_center += pm1->verts[i];
	}
	pm1->_center /= float(pm1->verts_size);
	pm1->_init_center = pm1->_center;

	PolyCollMesh* pm2 = new PolyCollMesh(pm1);

	if (dynm) {
		dmeshes->push_back(pm1);
		dmeshes_future->push_back(pm2);
		dd_ccache.resize(dd_ccache.size() + 1);
		dd_ccache[dd_ccache.size() - 1] = std::vector<CollCache>(dmeshes->size() - 1);
		for (int i = 0; i < dmeshes->size() - 1; i++) {
			dd_ccache[dd_ccache.size() - 1][i] = get_sep_plane((*dmeshes)[dd_ccache.size() - 1], (*dmeshes)[i]);
		}
		dl_ccache.resize(dl_ccache.size() + 1);
		dl_ccache[dl_ccache.size() - 1] = std::vector<CollCache>(lmeshes->size());
		for (int i = 0; i < lmeshes->size(); i++) {
			dl_ccache[dl_ccache.size() - 1][i] = get_sep_plane((*lmeshes)[i], (*dmeshes)[dl_ccache.size() - 1]);
		}
	}
	else {
		lmeshes->push_back(pm1);
		lmeshes_future->push_back(pm2);
		for (int i = 0; i < dmeshes->size(); i++) {
			dl_ccache[i].resize(lmeshes->size());
			dl_ccache[i][lmeshes->size() - 1] = get_sep_plane((*lmeshes)[lmeshes->size() - 1], (*dmeshes)[i]);
		}
	}
	new_mesh_lock.unlock();
}

void PrismPhysics::gen_and_add_pcmesh(std::vector<glm::vec3> plane_points, float face_thickness, float face_friction, bool dynm)
{
	new_mesh_lock.lock();
	PolyCollMesh* pm1 = new PolyCollMesh();

	pm1->face_epsilon = face_thickness;
	pm1->friction = face_friction;

	pm1->verts_size = plane_points.size();
	pm1->verts.resize(pm1->verts_size);
	memcpy(pm1->verts.data(), plane_points.data(), pm1->verts_size * sizeof(glm::vec3));

	pm1->faces_size = 1;
	pm1->faces.resize(pm1->faces_size);
	pm1->faces[0].vinds_size = pm1->verts_size;
	pm1->faces[0].vinds.resize(pm1->faces[0].vinds_size);

	pm1->edges_size = pm1->verts_size;
	pm1->edges.resize(pm1->edges_size);

	for (int i = 0; i < (pm1->verts_size); i++) {
		pm1->faces[0].vinds[i] = i;
		pm1->edges[i] = glm::ivec2(i, (i + 1) % pm1->edges_size);
	}

	pm1->faces[0].process_plane(&pm1->verts);

	for (int i = 0; i < pm1->verts_size; i++) {
		pm1->_center += pm1->verts[i];
	}

	glm::vec3 uaxn = glm::normalize(plane_points[1] - plane_points[0]);
	glm::vec3 vaxn = (plane_points[2] - plane_points[0]);
	vaxn = glm::normalize(vaxn - glm::dot(uaxn, vaxn) * uaxn);

	pm1->_uvw = glm::mat3(uaxn, vaxn, glm::cross(uaxn, vaxn));

	pm1->_center /= float(pm1->verts_size);
	pm1->_init_center = pm1->_center;

	PolyCollMesh* pm2 = new PolyCollMesh(pm1);

	if (dynm) {
		dmeshes->push_back(pm1);
		dmeshes_future->push_back(pm2);
		dd_ccache.resize(dd_ccache.size() + 1);
		dd_ccache[dd_ccache.size() - 1] = std::vector<CollCache>(dmeshes->size() - 1);
		for (int i = 0; i < dmeshes->size() - 1; i++) {
			dd_ccache[dd_ccache.size() - 1][i] = get_sep_plane((*dmeshes)[dd_ccache.size() - 1], (*dmeshes)[i]);
		}
		dl_ccache.resize(dl_ccache.size() + 1);
		dl_ccache[dl_ccache.size() - 1] = std::vector<CollCache>(lmeshes->size());
		for (int i = 0; i < lmeshes->size(); i++) {
			dl_ccache[dl_ccache.size() - 1][i] = get_sep_plane((*lmeshes)[i], (*dmeshes)[dl_ccache.size() - 1]);
		}
	}
	else {
		lmeshes->push_back(pm1);
		lmeshes_future->push_back(pm2);
		for (int i = 0; i < dmeshes->size(); i++) {
			dl_ccache[i].resize(lmeshes->size());
			dl_ccache[i][lmeshes->size() - 1] = get_sep_plane((*lmeshes)[lmeshes->size() - 1], (*dmeshes)[i]);
		}
	}
	new_mesh_lock.unlock();
}

CollCache PrismPhysics::get_sep_plane(PolyCollMesh* pm1, PolyCollMesh* pm2)
{

	CollCache outdata;
	bool found = true;

	for (int i = 0; i < pm1->faces_size; i++) {
		found = true;
		for (int j = 0; j < pm2->verts_size; j++) {
			float dist = glm::dot(pm1->faces[i].equation, glm::vec4(pm2->verts[j], 1));
			if (dist <= pm1->face_epsilon) {
				found = false;
				break;
			}
		}
		if (found) {
			CollCache outdata;
			outdata.sep_plane = pm1->faces[i].equation;
			outdata._dir = glm::normalize(glm::vec3(outdata.sep_plane));
			outdata.m1side = true;
			outdata.m2side = false;
			outdata.sep_plane_idx = i;
			return outdata;
		}
	}


	for (int i = 0; i < pm2->faces_size; i++) {
		found = true;
		for (int j = 0; j < pm1->verts_size; j++) {
			float dist = glm::dot(pm2->faces[i].equation, glm::vec4(pm1->verts[j], 1));
			if (dist <= pm2->face_epsilon) {
				found = false;
				break;
			}
		}
		if (found) {
			outdata.sep_plane = pm2->faces[i].equation;
			outdata._dir = glm::normalize(glm::vec3(outdata.sep_plane));
			outdata.m1side = false;
			outdata.m2side = true;
			outdata.sep_plane_idx = i;
			return outdata;
		}
	}

	size_t min_p_count = pm2->verts_size;
	size_t min_p_idx = 0;
	for (size_t i = 0; i < pm1->faces_size; i++) {
		size_t in_p = 0;
		for (size_t j = 0; j < pm2->verts_size; j++) {
			float dist = glm::dot(pm1->faces[i].equation, glm::vec4(pm2->verts[j], 1));
			if (dist <= pm1->face_epsilon) {
				in_p++;
			}
		}
		if (in_p < min_p_count) {
			min_p_count = in_p;
			min_p_idx = i;
		}
	}

	outdata.sep_plane = pm1->faces[min_p_idx].equation;
	outdata._dir = glm::normalize(glm::vec3(outdata.sep_plane));
	outdata.m1side = false;
	outdata.m2side = false;
	outdata.sep_plane_idx = min_p_idx;
	return outdata;
}

void PrismPhysics::run_physics(int rt_ms)
{
	for (int i = 0; i < rt_ms; i++) {
		run_physics_one_step();
	}
}

void PrismPhysics::advance_mesh_one_step(PolyCollMesh* pm)
{
	if (pm->running_anims.size() == 0) {
		glm::vec3 ldisp = (pm->_bvel * 0.001f) + (0.5f * 0.001f * 0.001f * pm->_bacc);
		pm->apply_displacement(ldisp);
		pm->_bvel += pm->_bacc * 0.001f;
	}
	else {
		pm->updateAnim(pm->running_anims[0], 1);
	}
}

glm::vec3 apply_dbounds_vel(glm::vec3 vec_to_bound, std::vector<DynBound> dbounds) {
	glm::vec3 vtc = glm::vec3(0);
	if (dbounds.size() == 0) return vec_to_bound;
	//if (glm::length(vec_to_bound) == 0) return glm::vec3(0);
	bool found_bounded = true;
	for (int cdi = 0; cdi < dbounds.size(); cdi++) {
		if (glm::dot(vec_to_bound, dbounds[cdi]._dir) < glm::dot(dbounds[cdi]._dir, dbounds[cdi]._vel)) {
			found_bounded = false;
			break;
		}
	}
	if (found_bounded) {
		return vec_to_bound;
	}

	glm::vec3 best_vtc = glm::vec3(0);
	int best_vtc_state = 0;

	for (int cdi = 0; cdi < dbounds.size(); cdi++) {
		if (std::isnan(dbounds[cdi]._dir.x)) dbounds[cdi]._dir = glm::vec3(0, 1, 0);
		float vcomp = glm::dot(vec_to_bound - dbounds[cdi]._vel, dbounds[cdi]._dir);
		//if (vcomp > 0) continue;
		for (int cdj = 0; cdj < dbounds.size(); cdj++) {
			if (glm::dot(dbounds[cdi]._vel, dbounds[cdi]._dir) < glm::dot(dbounds[cdj]._vel, dbounds[cdj]._dir) * glm::dot(dbounds[cdj]._dir, dbounds[cdi]._dir) && vcomp > 0) {
				continue;
			}
		}

		glm::vec3 tvel = normalize_vec_in_dir(vec_to_bound, dbounds[cdi]._plane, 0);
		float vtang = glm::length(tvel);
		float gtv = 0;
		float ltv = 0;

		std::vector<DynBound> cbounds;

		bool found_t = true;
		for (int cdj = 0; cdj < dbounds.size(); cdj++) {
			glm::vec3 tmpva = normalize_vec_in_dir(glm::dot(dbounds[cdj]._dir, dbounds[cdj]._vel - dbounds[cdi]._vel) * dbounds[cdj]._dir, dbounds[cdi]._dir, 0);
			if (glm::length(tmpva) > 0.00001) {
				float cos_th = glm::dot(glm::normalize(tmpva), glm::normalize(dbounds[cdj]._dir));
				DynBound ttdb = dbounds[cdj];
				ttdb._vel = glm::normalize(tmpva) * glm::dot(dbounds[cdj]._vel - dbounds[cdi]._vel, dbounds[cdj]._dir) / cos_th;
				ttdb._dir = glm::normalize(ttdb._vel);
				if (glm::dot(tvel - ttdb._vel, ttdb._dir) > 0 && glm::dot(dbounds[cdi]._dir, dbounds[cdj]._dir) < 0) {
					cbounds.push_back(ttdb);
				}
				else {
					found_t = false;
					break;
				}

			}
			else {
				tmpva = normalize_vec_in_dir(dbounds[cdj]._plane, dbounds[cdi]._plane, 0);
				if (glm::length(tmpva) > 0.00001) {
					DynBound ttdb = dbounds[cdj];
					ttdb._vel = glm::vec3(0);
					ttdb._dir = glm::normalize(tmpva);
					if (glm::dot(dbounds[cdi]._plane, dbounds[cdj]._plane) < 0) {
						cbounds.push_back(ttdb);
					}
					else {
						if (glm::dot(tvel - ttdb._vel, ttdb._dir) < 0) {
							found_t = false;
							break;
						}
					}
				}
			}
		}
		if (!found_t) continue;


		glm::vec3 tdvel = normalize_vec_in_dir(vec_to_bound, dbounds[cdi]._plane, 0);
		glm::vec3 best_tdvel = tdvel;
		for (int cbi = 0; cbi < cbounds.size(); cbi++) {
			float tcomp = glm::dot(tvel, cbounds[cbi]._dir);
			glm::vec3 tvel1 = normalize_vec_in_dir(tdvel, cbounds[cbi]._dir);
			glm::vec3 nvel1 = glm::dot(cbounds[cbi]._vel, cbounds[cbi]._dir) * cbounds[cbi]._dir;
			if (glm::length(tvel1) == 0) {
				best_tdvel = glm::vec3(0);
				continue;
			}
			glm::vec3 tvel1dir = glm::normalize(tvel1);
			for (int cbj = 0; cbj < cbounds.size(); cbj++) {
				if (cbj != cbi) {
					glm::vec3 tvel2 = glm::dot(cbounds[cbj]._vel, tvel1dir) * tvel1dir;
					glm::vec3 fvel2 = tvel2 + nvel1;
					for (int cbk = 0; cbk < cbounds.size(); cbk++) {
						if (glm::dot(fvel2 - cbounds[cbk]._vel, cbounds[cbk]._dir) < 0) {
							found_t = false;
							break;
						}
					}
					if (found_t) {
						best_tdvel = fvel2;
						break;
					}
				}
			}
			if (cbounds.size() == 1) {
				best_tdvel = tvel1 + nvel1;
				//found_t = false;
			}
		}

		if (found_t) {
			found_bounded = true;
			vtc = glm::dot(dbounds[cdi]._vel, dbounds[cdi]._dir) * dbounds[cdi]._dir + best_tdvel;
			break;
		}
		else {
			continue;
		}
	}

	return vtc;
}

glm::vec3 apply_dbounds_acc(glm::vec3 vec_to_bound, std::vector<DynBound> dbounds)
{
	glm::vec3 vtc = glm::vec3(0);
	if (dbounds.size() == 0) return vec_to_bound;
	//if (glm::length(vec_to_bound) == 0) return glm::vec3(0);
	bool found_bounded = true;
	for (int cdi = 0; cdi < dbounds.size(); cdi++) {
		if (glm::dot(vec_to_bound, dbounds[cdi]._dir) < glm::dot(dbounds[cdi]._dir, dbounds[cdi]._acc)) {
			found_bounded = false;
		}
	}
	if (found_bounded) {
		return vec_to_bound;
	}
	for (int cdi = 0; cdi < dbounds.size(); cdi++) {
		if (std::isnan(dbounds[cdi]._dir.x)) dbounds[cdi]._dir = glm::vec3(0, 1, 0);
		float vcomp = glm::dot(vec_to_bound - dbounds[cdi]._acc, dbounds[cdi]._dir);
		//if (vcomp > 0) continue;

		glm::vec3 tvel = normalize_vec_in_dir(vec_to_bound, dbounds[cdi]._acc, 0);
		float vtang = glm::length(tvel);
		float gtv = 0;
		float ltv = 0;

		std::vector<DynBound> cbounds;

		for (int cdj = 0; cdj < dbounds.size(); cdj++) {
			glm::vec3 tmpva = normalize_vec_in_dir(glm::dot(dbounds[cdj]._dir, dbounds[cdj]._acc) * dbounds[cdj]._dir, dbounds[cdi]._dir, 0);
			if (glm::length(tmpva) > 0) {
				float cos_th = glm::dot(glm::normalize(tmpva), glm::normalize(dbounds[cdj]._dir));
				DynBound ttdb = dbounds[cdj];
				ttdb._acc = glm::normalize(tmpva) * glm::dot(dbounds[cdj]._acc, dbounds[cdj]._dir) / cos_th;
				ttdb._dir = glm::normalize(ttdb._acc);
				cbounds.push_back(ttdb);
			}
			else {
				tmpva = normalize_vec_in_dir(dbounds[cdj]._plane, dbounds[cdi]._plane, 0);
				if (glm::length(tmpva) > 0) {
					DynBound ttdb = dbounds[cdj];
					ttdb._acc = glm::vec3(0);
					ttdb._dir = glm::normalize(tmpva);
					cbounds.push_back(ttdb);
				}
			}
		}

		bool found_t = true;
		glm::vec3 tdvel = normalize_vec_in_dir(vec_to_bound, dbounds[cdi]._plane, 0);
		glm::vec3 best_tdvel = tdvel;
		for (int cbi = 0; cbi < cbounds.size(); cbi++) {
			float tcomp = glm::dot(tvel, cbounds[cbi]._dir);
			glm::vec3 tvel1 = normalize_vec_in_dir(tdvel, cbounds[cbi]._dir);
			glm::vec3 nvel1 = glm::dot(cbounds[cbi]._acc, cbounds[cbi]._dir) * cbounds[cbi]._dir;
			if (glm::length(tvel1) == 0) {
				best_tdvel = glm::vec3(0);
				continue;
			}
			glm::vec3 tvel1dir = glm::normalize(tvel1);
			for (int cbj = 0; cbj < cbounds.size(); cbj++) {
				if (cbj != cdi) {
					glm::vec3 tvel2 = glm::dot(cbounds[cbj]._acc, tvel1dir) * tvel1dir;
					glm::vec3 fvel2 = tvel2 + nvel1;
					for (int cbk = 0; cbk < cbounds.size(); cbk++) {
						if (glm::dot(fvel2 - cbounds[cbk]._acc, cbounds[cbk]._dir) < 0) {
							found_t = false;
							break;
						}
					}
					if (found_t) {
						best_tdvel = fvel2;
						break;
					}
				}
			}
			if (cbounds.size() == 1) {
				best_tdvel = tvel1 + nvel1;
			}
		}

		if (found_t) {
			found_bounded = true;
			vtc = glm::dot(dbounds[cdi]._acc, dbounds[cdi]._dir) * dbounds[cdi]._dir + best_tdvel;
			break;
		}
		else {
			continue;
		}
	}

	return vtc;
}

void PrismPhysics::run_physics_one_step()
{
	new_mesh_lock.lock();
	for (size_t i = 0; i < lmeshes->size(); i++) {
		thread_pool->add_task(&advance_mesh_one_step, lmeshes_future->at(i));
	}

	std::vector<std::vector<DynBound>> dbounds = std::vector<std::vector<DynBound>>(dmeshes->size());
	std::vector<std::vector<DynBound>> fric_dbounds = std::vector<std::vector<DynBound>>(dmeshes->size());
	
	thread_pool->wait_till_done();

	for (size_t i = 0; i < dmeshes->size(); i++) {
		(*dmeshes_future)[i]->_bvel = (*dmeshes_future)[i]->_vel;
		(*dmeshes_future)[i]->_bacc = (*dmeshes_future)[i]->_acc;
		advance_mesh_one_step((*dmeshes_future)[i]);
		(*dmeshes_future)[i]->_vel = (*dmeshes_future)[i]->_bvel;
		for (int j = 0; j < lmeshes->size(); j++) {
			CollCache tmp_cc = dl_ccache[i][j];
			bool spl_invalid = false;
			glm::vec4 tmp_spl;
			if (!tmp_cc.m1side && !tmp_cc.m2side) {
				spl_invalid = true;
			}
			else {
				if (tmp_cc.m1side) {
					tmp_spl = lmeshes_future->at(j)->faces[tmp_cc.sep_plane_idx].equation;
					for (int k = 0; k < dmeshes_future->at(i)->verts_size; k++) {
						if (glm::dot(tmp_spl, glm::vec4(dmeshes_future->at(i)->verts[k], 1)) <= lmeshes_future->at(j)->face_epsilon) {
							spl_invalid = true;
							break;
						}
					}
				}
				if (tmp_cc.m2side) {
					tmp_spl = dmeshes_future->at(i)->faces[tmp_cc.sep_plane_idx].equation;
					for (int k = 0; k < lmeshes_future->at(j)->verts_size; k++) {
						if (glm::dot(tmp_spl, glm::vec4(lmeshes_future->at(j)->verts[k], 1)) <= dmeshes_future->at(i)->face_epsilon) {
							spl_invalid = true;
							break;
						}
					}
				}
			}
			
			if (spl_invalid) {
				CollCache new_cc = get_sep_plane((*lmeshes_future)[j], (*dmeshes_future)[i]);
				if (!(new_cc.m1side || new_cc.m2side)) {
					if (lmeshes_future->at(j)->coll_behav == "physics") {
						DynBound tmp_db;
						tmp_db._plane = (glm::dot(glm::vec4((*dmeshes)[i]->_center, 1), tmp_cc.sep_plane) > 0) ? tmp_cc.sep_plane : -tmp_cc.sep_plane;
						tmp_db._dir = glm::vec3(tmp_db._plane);
						tmp_db._vel = (*lmeshes_future)[j]->_vel;
						tmp_db._acc = (*lmeshes_future)[j]->_acc;
						tmp_db.friction = (*lmeshes_future)[j]->friction;
						tmp_db.epsilon = (*lmeshes_future)[j]->face_epsilon;
						dbounds[i].push_back(tmp_db);

						tmp_db._vel = glm::vec3(0);
						tmp_db._acc = glm::vec3(0);
						fric_dbounds[i].push_back(tmp_db);
					}
					else if (lmeshes_future->at(j)->coll_behav == "animself") {
						if (lmeshes_future->at(j)->coll_behav_args.size() > 0 && lmeshes_future->at(j)->running_anims.size() == 0) {
							lmeshes_future->at(j)->running_anims.push_back(lmeshes_future->at(j)->coll_behav_args[0]);
						}
						DynBound tmp_db;
						tmp_db._plane = (glm::dot(glm::vec4((*dmeshes)[i]->_center, 1), tmp_cc.sep_plane) > 0) ? tmp_cc.sep_plane : -tmp_cc.sep_plane;
						tmp_db._dir = glm::vec3(tmp_db._plane);
						tmp_db._vel = (*lmeshes_future)[j]->_vel;
						tmp_db._acc = (*lmeshes_future)[j]->_acc;
						tmp_db.friction = (*lmeshes_future)[j]->friction;
						tmp_db.epsilon = (*lmeshes_future)[j]->face_epsilon;
						dbounds[i].push_back(tmp_db);

						tmp_db._vel = glm::vec3(0);
						tmp_db._acc = glm::vec3(0);
						fric_dbounds[i].push_back(tmp_db);
					}
					else if (lmeshes_future->at(j)->coll_behav == "animremote") {
						if (lmeshes_future->at(j)->coll_behav_args.size() > 1) {
							int emi = stoi(lmeshes_future->at(j)->coll_behav_args[1]);
							if (lmeshes_future->at(emi)->running_anims.size() == 0) {
								lmeshes_future->at(emi)->running_anims.push_back(lmeshes_future->at(j)->coll_behav_args[0]);
							}
						}
						DynBound tmp_db;
						tmp_db._plane = (glm::dot(glm::vec4((*dmeshes)[i]->_center, 1), tmp_cc.sep_plane) > 0) ? tmp_cc.sep_plane : -tmp_cc.sep_plane;
						tmp_db._dir = glm::vec3(tmp_db._plane);
						tmp_db._vel = (*lmeshes_future)[j]->_vel;
						tmp_db._acc = (*lmeshes_future)[j]->_acc;
						tmp_db.friction = (*lmeshes_future)[j]->friction;
						tmp_db.epsilon = (*lmeshes_future)[j]->face_epsilon;
						dbounds[i].push_back(tmp_db);

						tmp_db._vel = glm::vec3(0);
						tmp_db._acc = glm::vec3(0);
						fric_dbounds[i].push_back(tmp_db);
					}
					else if (lmeshes_future->at(j)->coll_behav == "kill" && i==0) {
						glm::vec3 rspn_tran = glm::vec3(0);
						rspn_tran = glm::vec3(10, 10, 10) - dmeshes_future->at(0)->_center;
						dmeshes_future->at(0)->apply_displacement(rspn_tran);
						dmeshes_future->at(0)->_vel = glm::vec3(0);
						dmeshes_future->at(0)->_bvel = glm::vec3(0);

						rspn_tran = glm::vec3(10, 10, 10) - dmeshes->at(0)->_center;
						dmeshes->at(0)->apply_displacement(rspn_tran);
						dmeshes->at(0)->_vel = glm::vec3(0);
						dmeshes->at(0)->_bvel = glm::vec3(0);

						for (int i = 0; i < dmeshes->size() - 1; i++) {
							dd_ccache[0][i] = get_sep_plane(dmeshes->at(0), dmeshes->at(i));
						}
						for (int i = 0; i < lmeshes->size(); i++) {
							dl_ccache[0][i] = get_sep_plane(lmeshes->at(i), dmeshes->at(0));
						}
						break;
					}
				}
				dl_ccache[i][j] = new_cc;
			}
		}
		if (dbounds[i].size() > 0) {
			for (int dbfi = 0; dbfi < dbounds[i].size(); dbfi++) {
				move_mesh_out_plane((*dmeshes_future)[i], dbounds[i][dbfi]._plane, dbounds[i][dbfi].epsilon);
			}
			(*dmeshes_future)[i]->_bvel = apply_dbounds_vel((*dmeshes_future)[i]->_vel, dbounds[i]);
			(*dmeshes_future)[i]->_bacc = apply_dbounds_vel((*dmeshes_future)[i]->_acc, dbounds[i]);


			if (!dmeshes_future->at(i)->controlled) {
				// Apply friction
				glm::vec3 friction = glm::vec3(0.0f);
				std::vector<glm::vec3> friction_list;
				std::vector<glm::vec3> friction_end_list;
				glm::vec3 min_frict_dvel = glm::vec3(0);
				for (size_t j = 0; j < dbounds[i].size(); j++) {
					if (glm::dot(dbounds[i][j]._dir, (*dmeshes_future)[i]->_acc - dbounds[i][j]._acc) < 0.0 &&
						glm::dot(dbounds[i][j]._dir, (*dmeshes_future)[i]->_bvel - dbounds[i][j]._vel) <= 0.0) {
						glm::vec3 rvel = ((*dmeshes_future)[i]->_bvel - dbounds[i][j]._vel);
						glm::vec3 racc = ((*dmeshes_future)[i]->_bacc - dbounds[i][j]._acc);
						glm::vec3 rtvel = normalize_vec_in_dir(rvel, dbounds[i][j]._dir, 0);
						glm::vec3 sur_fric = glm::vec3(0);
						if (glm::length(rtvel) > 0) {
							sur_fric = apply_dbounds_vel(
								normalize_vec_in_dir(-dbounds[i][j].friction * glm::normalize(rtvel), dbounds[i][j]._dir, 0),
								fric_dbounds[i]
							);
							if (glm::length(sur_fric) > 0) {
								float lftime = -glm::dot(sur_fric, rtvel) / glm::dot(sur_fric, sur_fric);
								if (lftime > 0 && lftime < 1) {
									friction_end_list.push_back(sur_fric);
								}
							}
						}
						else {
							if (glm::length(racc) <= dbounds[i][j].friction) {
								sur_fric = apply_dbounds_acc(-racc, fric_dbounds[i]);
							}
							else {
								if (glm::length(racc) > 0) {
									sur_fric = apply_dbounds_acc(
										normalize_vec_in_dir(-dbounds[i][j].friction * glm::normalize(racc), dbounds[i][j]._dir, 0),
										fric_dbounds[i]
									);
								}
							}
						}
						friction_list.push_back(sur_fric);
						friction += sur_fric;
					}
				}


				(*dmeshes_future)[i]->_vel = (*dmeshes_future)[i]->_bvel;
				(*dmeshes_future)[i]->_bacc += friction;

				//advance_mesh_one_step((*dmeshes_future)[i]);
				(*dmeshes_future)[i]->_vel = (*dmeshes_future)[i]->_bvel;

				for (size_t k = 0; k < friction_end_list.size(); k++) {
					(*dmeshes_future)[i]->_vel = normalize_vec_in_dir((*dmeshes_future)[i]->_vel, friction_end_list[k], 0);
				}
			}
			else {
				(*dmeshes_future)[i]->_vel = (*dmeshes_future)[i]->_bvel;
				//advance_mesh_one_step((*dmeshes_future)[i]);
				(*dmeshes_future)[i]->_vel = (*dmeshes_future)[i]->_bvel;
			}
			
		}
	}

	for (int i = 0; i < lmeshes->size(); i++) {
		(*(*lmeshes)[i]) = (*(*lmeshes_future)[i]);
	}
	for (int i = 0; i < dmeshes->size(); i++) {
		(*(*dmeshes)[i]) = (*(*dmeshes_future)[i]);
	}

	new_mesh_lock.unlock();
}

DynamicCollPoint PrismPhysics::find_ray_first_coll(glm::vec3 raystart, glm::vec3 raydir)
{
	DynamicCollPoint cpt;
	cpt.time = 100000;
	cpt.time = 100000;
	for (int i = 0; i < lmeshes->size(); i++) {
		CollPoint tmp = lmeshes->at(i)->find_ray_first_coll(raystart, raydir);
		if (tmp.will_collide && tmp.time < cpt.time) {
			cpt.will_collide = true;
			cpt.relative_disp = lmeshes->at(i)->_uvw * (tmp.displacement1 - lmeshes->at(i)->_center);
			cpt.obj_index = i;
			cpt.time = tmp.time;
		}
	}
	return cpt;
}