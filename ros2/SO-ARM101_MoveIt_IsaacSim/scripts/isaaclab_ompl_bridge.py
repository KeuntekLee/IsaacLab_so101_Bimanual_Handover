import argparse
import math
import threading
from dataclasses import dataclass
from typing import Iterable, List, Optional

import rclpy
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes, MotionPlanRequest
from moveit_msgs.srv import GetMotionPlan
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory

DEFAULT_ARM_JOINTS = [
	"Rotation",
	"Pitch",
	"Elbow",
	"Wrist_Pitch",
	"Wrist_Roll",
]

MOVEIT_ERROR_NAMES = {
	MoveItErrorCodes.SUCCESS: "SUCCESS",
	MoveItErrorCodes.FAILURE: "FAILURE",
	MoveItErrorCodes.PLANNING_FAILED: "PLANNING_FAILED",
	MoveItErrorCodes.INVALID_MOTION_PLAN: "INVALID_MOTION_PLAN",
	MoveItErrorCodes.MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE: (
		"MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE"
	),
	MoveItErrorCodes.CONTROL_FAILED: "CONTROL_FAILED",
	MoveItErrorCodes.UNABLE_TO_AQUIRE_SENSOR_DATA: "UNABLE_TO_AQUIRE_SENSOR_DATA",
	MoveItErrorCodes.TIMED_OUT: "TIMED_OUT",
	MoveItErrorCodes.PREEMPTED: "PREEMPTED",
	MoveItErrorCodes.START_STATE_IN_COLLISION: "START_STATE_IN_COLLISION",
	MoveItErrorCodes.START_STATE_VIOLATES_PATH_CONSTRAINTS: (
		"START_STATE_VIOLATES_PATH_CONSTRAINTS"
	),
	MoveItErrorCodes.START_STATE_INVALID: "START_STATE_INVALID",
	MoveItErrorCodes.GOAL_IN_COLLISION: "GOAL_IN_COLLISION",
	MoveItErrorCodes.GOAL_VIOLATES_PATH_CONSTRAINTS: "GOAL_VIOLATES_PATH_CONSTRAINTS",
	MoveItErrorCodes.GOAL_CONSTRAINTS_VIOLATED: "GOAL_CONSTRAINTS_VIOLATED",
	MoveItErrorCodes.GOAL_STATE_INVALID: "GOAL_STATE_INVALID",
	MoveItErrorCodes.UNRECOGNIZED_GOAL_TYPE: "UNRECOGNIZED_GOAL_TYPE",
	MoveItErrorCodes.INVALID_GROUP_NAME: "INVALID_GROUP_NAME",
	MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS: "INVALID_GOAL_CONSTRAINTS",
	MoveItErrorCodes.INVALID_ROBOT_STATE: "INVALID_ROBOT_STATE",
	MoveItErrorCodes.INVALID_LINK_NAME: "INVALID_LINK_NAME",
	MoveItErrorCodes.INVALID_OBJECT_NAME: "INVALID_OBJECT_NAME",
	MoveItErrorCodes.FRAME_TRANSFORM_FAILURE: "FRAME_TRANSFORM_FAILURE",
	MoveItErrorCodes.COLLISION_CHECKING_UNAVAILABLE: "COLLISION_CHECKING_UNAVAILABLE",
	MoveItErrorCodes.ROBOT_STATE_STALE: "ROBOT_STATE_STALE",
	MoveItErrorCodes.SENSOR_INFO_STALE: "SENSOR_INFO_STALE",
	MoveItErrorCodes.COMMUNICATION_FAILURE: "COMMUNICATION_FAILURE",
	MoveItErrorCodes.CRASH: "CRASH",
	MoveItErrorCodes.ABORT: "ABORT",
	MoveItErrorCodes.NO_IK_SOLUTION: "NO_IK_SOLUTION",
}
@dataclass
class JointPlanInput:
	source: str
	joint_names: List[str]
	goal_positions: List[float]
	start_positions: Optional[List[float]]
	

class IsaacLabOmplBridge(Node):
	def __init__(self, args:argparse.Namespace) -> None:
		super().__init__(
			"isaaclab_ompl_bridge",
			parameter_overrides=[Parameter("use_sim_time", value=args.use_sim_time)],
			automatically_declare_parameters_from_overrides=True,
		)
		
		self._callback_group = ReentrantCallbackGroup()
		self._lock = threading.Lock()
		self._planning_active = False
		self._request_id = 0
		
		self._joint_names = args.joint_names
		self._moveit_service = args.moveit_service
		self._group_name = args.group_name
		self._pipeline_id = args.pipeline_id
		self._planner_id = args.planner_id
		self._goal_tolerance = args.goal_tolerance
		self._allowed_planning_time = args.allowed_planning_time
		self._num_planning_attempts = args.num_planning_attempts
		self._max_velocity_scaling_factor = args.max_velocity_scaling_factor
		self._max_acceleration_scaling_factor = args.max_acceleration_scaling_factor
		self._wait_for_moveit_timeout_sec = args.wait_for_moveit_timeout_sec
		
		self._moveit_client = self.create_client(
			GetMotionPlan,
			self._moveit_service,
			callback_group=self._callback_group,
		)
		self._trajectory_pub = self.create_publisher(
			JointTrajectory,
			args.planned_trajectory_topic,
			10,
		)
		self._status_pub = self.create_publisher(String, args.plan_status_topic, 10)
		self.create_subscription(
			Float64MultiArray,
			args.joint_goal_topic,
			self._on_joint_goal,
			10,
			callback_group=self._callback_group,
		)
		self.create_subscription(
			JointTrajectory,
			args.plan_request_topic,
			self._on_plan_request,
			10,
			callback_group=self._callback_group,
		)
		self._publish_status(
			"READY "
			f"joint_goal_topic={args.joint_goal_topic} "
			f"plan_request_topic={args.plan_request_topic} "
			f"planned_trajectory_topic={args.planned_trajectory_topic} "
			f"moveit_service={self._moveit_service} "
		)
	def _on_joint_goal(self, msg: Float64MultiArray) -> None:
		values = [float(value) for value in msg.data]
		joint_count = len(self._joint_names)
		self.get_logger().info(
			f"RECEIVED joint_goal len={len(values)} raw={self._format_values(values)}"
		)
		
		if len(values) == joint_count:
			plan_input = JointPlanInput(
				source="Float64MultiArray(goal)",
				joint_names=self._joint_names,
				start_positions=None,
				goal_positions=values,
			)
		elif len(values) == 2 * joint_count:
			plan_input = JointPlanInput(
				source="Float64MultiArray(start+goal)",
				joint_names=self._joint_names,
				start_positions=values[:joint_count],
				goal_positions=values[joint_count:],
			)
		else:
			self._publish_status(
				"INVALID_REQUEST "
				f"/isaaclab/joint_goal expects {joint_count} goal values or "
				f"{2 * joint_count} start+goal values, got {len(values)}"
			)
			return
			
		self._submit_plan(plan_input)
		
	def _on_plan_request(self, msg: JointTrajectory) -> None:
		if not msg.points:
			self._publish_status("INVALID_REQUEST /isaaclab/plan_request has no points")
			return
			
		joint_names = list(msg.joint_names) if msg.joint_names else self._joint_names
		if len(msg.points) == 1:
			start_positions = None
			goal_positions = list(msg.points[0].positions)
		else:
			start_positions = list(msg.points[0].positions)
			goal_positions = list(msg.points[-1].positions)
		self.get_logger().info(
			"RECEIVED plan_request "
			f"points={len(msg.points)} joints={joint_names} "
			f"start={self._format_values(start_positions)} "
			f"goal={self._format_values(goal_positions)} "
		)
		
		self._submit_plan(
			JointPlanInput(
				source="JointTrajectory",
				joint_names=joint_names,
				start_positions=start_positions,
				goal_positions=goal_positions,
			)
		)
	
	def _format_values(self, values: Optional[Iterable[float]], precision: int = 4) -> str:
		if values is None:
			return "None"
		return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"
	
	def _submit_plan(self, plan_input: JointPlanInput) -> None:
		validation_error = self._validate_input(plan_input)
		if validation_error:
			self._publish_status(f"INVALID_REQUEST {validation_error}")
			return
			
		with self._lock:
			if self._planning_active:
				self._publish_status("BUSY previous planning request is still running")
				return
			self._planning_active = True
			self._request_id += 1
			request_id = self._request_id
			
		if not self._moveit_client.service_is_ready():
			self._publish_status(f"WAITING_FOR_MOVEIT request_id={request_id}")
			if not self._moveit_client.wait_for_service(
				timeout_sec=self._wait_for_moveit_timeout_sec
			):
				self._finish_request()
				self._publish_status(
					f"FAILED request_id={request_id} error=moveit_service_unavailable "
					f"service={self._moveit_service}"
				)
				return
				
		request = self._build_motion_plan_request(plan_input)
		self._publish_status(
			f"PLANNING request_id={request_id} source={plan_input.source} "
			f"start={'provided' if plan_input.start_positions else 'current_state'}"
		)
		future = self._moveit_client.call_async(request)
		future.add_done_callback(
			lambda completed_future: self._on_moveit_response(
				completed_future,
				request_id,
				plan_input.source,
			)
		)
		
	def _build_motion_plan_request(
		self,
		plan_input: JointPlanInput,
	) -> GetMotionPlan.Request:
		motion_request = MotionPlanRequest()
		motion_request.group_name = self._group_name
		motion_request.pipeline_id = self._pipeline_id
		motion_request.planner_id = self._planner_id
		motion_request.num_planning_attempts = self._num_planning_attempts
		motion_request.allowed_planning_time = self._allowed_planning_time
		motion_request.max_velocity_scaling_factor = self._max_velocity_scaling_factor
		motion_request.max_acceleration_scaling_factor = (
			self._max_acceleration_scaling_factor
		)
		
		motion_request.start_state.is_diff = True
		if plan_input.start_positions is not None:
			motion_request.start_state.joint_state = self._joint_state(
				plan_input.joint_names,
				plan_input.start_positions,
			)
			
		motion_request.goal_constraints = [
			self._joint_goal_constraints(
				plan_input.joint_names,
				plan_input.goal_positions,
				self._goal_tolerance,
			)
		]
		
		request = GetMotionPlan.Request()
		request.motion_plan_request = motion_request
		return request
		
	def _on_moveit_response(self, future, request_id: int, source: str) -> None:
		try:
			response = future.result()
		except Exception as exc:
			self._finish_request()
			self._publish_status(f"FAILED request_id={request_id} error={exc}")
			return
			
		self._finish_request()
		
		plan_response = response.motion_plan_response
		error_code = plan_response.error_code.val
		error_name = MOVEIT_ERROR_NAMES.get(error_code, f"UNKNOWN_{error_code}")
		trajectory = plan_response.trajectory.joint_trajectory
		
		if error_code == MoveItErrorCodes.SUCCESS:
			trajectory.header.stamp = self.get_clock().now().to_msg()
			self._trajectory_pub.publish(trajectory)
			self._publish_status(
				f"SUCCESS request_id={request_id} source={source} "
				f"points={len(trajectory.points)} "
				f"planning_time={plan_response.planning_time:.4f}"
			)
		else:
			self._publish_status(
				f"FAILED request_id={request_id} source={source} "
				f"error_code={error_code} error={error_name}"
			)
			
	def _finish_request(self) -> None:
		with self._lock:
			self._planning_active = False
			
	def _validate_input(self, plan_input: JointPlanInput) -> str:
		joint_count = len(plan_input.joint_names)
		if joint_count == 0:
			return "joint_names in empty"
		if len(plan_input.goal_positions) != joint_count:
			return (
				f"goal length {len(plan_input.goal_positions)} does not match "
				f"joint_names length {joint_count}"
			)
		if (
			plan_input.start_positions is not None
			and len(plan_input.start_positions) != joint_count
		):
			return (
				f"start_length {len(plan_input.start_positions)} does not match "
				f"joint_names length {joint_count}"
			)
			
		values = list(plan_input.goal_positions)
		if plan_input.start_positions is not None:
			values.extend(plan_input.start_positions)
		if not all(math.isfinite(value) for value in values):
			return "joint positions must be finite"
			
		return ""
		
	def _joint_state(
		self,
		joint_names: Iterable[str],
		positions: Iterable[float],
	) -> JointState:
		joint_state = JointState()
		joint_state.header.stamp = self.get_clock().now().to_msg()
		joint_state.name = list(joint_names)
		joint_state.position = list(positions)
		return joint_state
		
	@staticmethod
	def _joint_goal_constraints(
		joint_names: Iterable[str],
		positions: Iterable[float],
		tolerance: float,
	) -> Constraints:
		constraints = Constraints()
		constraints.name = "isaaclab_joint_goal"
		constraints.joint_constraints = [
			JointConstraint(
				joint_name=joint_name,
				position=position,
				tolerance_above=tolerance,
				tolerance_below=tolerance,
				weight=1.0,
			)
			for joint_name, position in zip(joint_names, positions)
		]
		return constraints
	
	def _publish_status(self, text: str) -> None:
		msg = String()
		msg.data = text
		self._status_pub.publish(msg)
		self.get_logger().info(text)
		
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Standalone IsaacLab to MoveIt OMPL bridge."
	)
	parser.add_argument("--use-sim-time", action="store_true")
	parser.add_argument("--moveit-service", default="/plan_kinematic_path")
	parser.add_argument("--joint-goal-topic", default="/isaaclab/joint_goal")
	parser.add_argument("--plan-request-topic", default="/isaaclab/plan_request")
	parser.add_argument("--planned-trajectory-topic", default="/isaaclab/planned_trajectory")
	parser.add_argument("--plan-status-topic", default="/isaaclab/plan_status")
	parser.add_argument("--group-name", default="arm")
	parser.add_argument("--pipeline-id", default="ompl")
	parser.add_argument("--planner-id", default="")
	parser.add_argument("--joint-names", nargs="+", default=DEFAULT_ARM_JOINTS)
	parser.add_argument("--goal-tolerance", type=float, default=0.001)
	parser.add_argument("--allowed-planning-time", type=float, default=5.0)
	parser.add_argument("--num-planning-attempts", type=int, default=5)
	parser.add_argument("--max-velocity-scaling-factor", type=float, default=0.1)
	parser.add_argument("--max-acceleration-scaling-factor", type=float, default=0.1)
	parser.add_argument("--wait-for-moveit-timeout-sec", type=float, default=3.0)
	return parser.parse_args()
	
def main() -> None:
	args = parse_args()
	rclpy.init()
	node = IsaacLabOmplBridge(args)
	executor = MultiThreadedExecutor(num_threads=2)
	executor.add_node(node)
	try:
		executor.spin()
	finally:
		executor.remove_node(node)
		node.destroy_node()
		if rclpy.ok():
			rclpy.shutdown()
			
			
if __name__ == "__main__":
	main()

			
