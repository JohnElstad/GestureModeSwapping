#!/usr/bin/env python
# license removed for brevity
import rospy
from geometry_msgs.msg import Twist

def main():
	pub_move = rospy.Publisher('AGBOT1_cmd_vel', Twist, queue_size=10)
	rospy.init_node('fake_bot_move', anonymous=True)
	rate = rospy.Rate(10) # 10hz
	cmd = Twist()
	rospy.loginfo("started")
	while not rospy.is_shutdown():
		
		linear_vel = 0.04
		angular_vel = 0.0

		cmd.linear.x = linear_vel
		cmd.angular.z = angular_vel

		pub_move.publish(cmd)
		rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
