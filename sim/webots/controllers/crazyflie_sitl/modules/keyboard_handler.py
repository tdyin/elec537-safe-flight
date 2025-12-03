"""
Keyboard input handler for manual control.

Processes keyboard input for manual drone control mode.
"""

from controller import Keyboard


class KeyboardHandler:
    """Handles keyboard input for manual control."""
    
    def __init__(self, robot, timestep):
        """
        Initialize keyboard handler.
        
        Args:
            robot: Webots robot instance
            timestep: Controller timestep
        """
        self.keyboard = Keyboard()
        self.keyboard.enable(timestep)
        
        # Control state
        self.manual_control = False
        
        # Manual control speeds
        self.manual_forward_speed = 0.5
        self.manual_sideways_speed = 0.5
        self.manual_vertical_speed = 0.3
        self.manual_yaw_rate = 0.5
    
    def process_input(self):
        """
        Process keyboard input for manual control.
        
        Returns:
            tuple: (vx, vy, height_change, yaw_rate) if manual mode, None if autonomous
        """
        vx = vy = yaw_rate = height_change = 0.0
        
        key = self.keyboard.getKey()
        while key > 0:
            # Toggle manual/autonomous mode
            if key == ord(' '):
                self.manual_control = not self.manual_control
                mode = "MANUAL" if self.manual_control else "AUTONOMOUS"
                print(f"\n{'='*60}\n🎮 {mode}\n{'='*60}")
            
            # Force autonomous mode
            elif key == ord('R'):
                if self.manual_control:
                    self.manual_control = False
                    print("\n🤖 AUTO mode")
            
            # Manual control commands
            if self.manual_control:
                if key == ord('W'):
                    vx += self.manual_forward_speed
                elif key == ord('S'):
                    vx -= self.manual_forward_speed
                elif key == ord('A'):
                    vy += self.manual_sideways_speed
                elif key == ord('D'):
                    vy -= self.manual_sideways_speed
                elif key == ord('Q'):
                    yaw_rate += self.manual_yaw_rate
                elif key == ord('E'):
                    yaw_rate -= self.manual_yaw_rate
                elif key == Keyboard.UP:
                    height_change += self.manual_vertical_speed
                elif key == Keyboard.DOWN:
                    height_change -= self.manual_vertical_speed
            
            key = self.keyboard.getKey()
        
        return (vx, vy, height_change, yaw_rate) if self.manual_control else None
    
    def is_manual_mode(self):
        """Check if in manual control mode."""
        return self.manual_control
