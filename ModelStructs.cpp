#include "ModelStructs.h"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>
#include <iostream>

static inline glm::mat4 makeTMatrix(glm::vec3 location, glm::mat4 rotate, glm::vec3 scale){
	return glm::translate(glm::mat4(1.0f), location) * rotate * glm::scale(glm::mat4(1.0f), scale);
}

glm::mat4 LRS::getTMatrix()
{
	return makeTMatrix(location, rotate, scale);
}

int BoneAnimData::getNextEventTime()
{
	return steps[curr_step].stepduration_ms - steps[curr_step].curr_time;
}

glm::vec3 BoneAnimData::getNextVelHint()
{
	return (steps[curr_step].finalPos - steps[curr_step].initPos) * 1000.0f / float(steps[curr_step].stepduration_ms);
}

LRS BoneAnimData::transformAfterGap(int gap_ms)
{
	LRS animLRS;

	while (gap_ms > 0) {
		int step_rem = steps[curr_step].stepduration_ms - steps[curr_step].curr_time;
		
		if (gap_ms >= step_rem) {
			int step_time = step_rem;
			steps[curr_step].curr_time = 0;

			float fract = float(step_time) / float(steps[curr_step].stepduration_ms);
			//float prog = float(steps[curr_step].curr_time) / float(steps[curr_step].stepduration_ms);
			animLRS.location = steps[curr_step].finalPos;
			animLRS.rotate = glm::rotate(
				glm::mat4(1),
				(1 - fract) * steps[curr_step].initAngle + fract * steps[curr_step].finalAngle,
				steps[curr_step].rotAxis);
			animLRS.scale = (1 - fract) * steps[curr_step].initScale + fract * steps[curr_step].finalScale;

			curr_step = (curr_step + 1) % steps.size();
			if (curr_step > steps.size() - 1) {
				curr_step = 0;
				curr_time = 0;
				break;
			}
			if (curr_step == 0 && !loop_anim) {
				animLRS.anim_finished = true;
				curr_time = 0;
				break;
			}
			curr_time += step_time;
			gap_ms -= step_time;
		}
		else {
			int step_time = gap_ms;
			steps[curr_step].curr_time += step_time;

			float fract = float(step_time) / float(steps[curr_step].stepduration_ms);
			float prog = float(steps[curr_step].curr_time) / float(steps[curr_step].stepduration_ms);
			animLRS.location = (1 - prog) * steps[curr_step].initPos + prog * steps[curr_step].finalPos;
			animLRS.rotate = glm::rotate(
				glm::mat4(1),
				(1 - fract) * steps[curr_step].initAngle + fract * steps[curr_step].finalAngle,
				steps[curr_step].rotAxis);
			animLRS.scale = (1 - fract) * steps[curr_step].initScale + fract * steps[curr_step].finalScale;

			curr_time += step_time;
			gap_ms -= step_time;
		}
	}
	return animLRS;
}

void PrismModelData::updateAnim(std::string animName, int gap_ms)
{
	LRS animLRS = anims[animName].transformAfterGap(gap_ms);
	objLRS.location += animLRS.location;
	objLRS.rotate *= animLRS.rotate;
	objLRS.scale = {
		objLRS.scale.x * animLRS.scale.x,
		objLRS.scale.y * animLRS.scale.y,
		objLRS.scale.z * animLRS.scale.z
	};
}
