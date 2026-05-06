from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import Gf, UsdGeom, Usd, Sdf
from scipy.spatial.transform import Rotation as R
import numpy as np

class Gripper:
    def __init__(self):
        self.grasped_object_path = None
        self.gripper_frame_path = None
        self.position_offest = None
        self.initial_object_rotation = None
        self.initial_gripper_rotation = None

    def reset():
        #TODO reset pick object
        return
        
    def add_object_to_gripper(self, object_path, gripper_frame_path):
        
        self.grasped_object_path = object_path
        self.gripper_frame_path = gripper_frame_path
                
        transform_prim = get_prim_at_path("/World/glass_rod")
        if not transform_prim.IsValid():
            raise ValueError(f"Object at path is not valid.")   

        self.inverse_transform_matrix = UsdGeom.Xformable(transform_prim).ComputeLocalToWorldTransform(0).GetInverse()
        
        # 保存初始的旋转关系
        object_prim = get_prim_at_path(object_path)
        gripper_prim = get_prim_at_path(gripper_frame_path)
        
        object_initial_transform = UsdGeom.Xformable(object_prim).ComputeLocalToWorldTransform(0)
        gripper_initial_transform = UsdGeom.Xformable(gripper_prim).ComputeLocalToWorldTransform(0)
        
        # 从旋转矩阵转换为四元数，然后创建 Rotation
        object_rot_matrix = object_initial_transform.ExtractRotationMatrix()
        gripper_rot_matrix = gripper_initial_transform.ExtractRotationMatrix()
        
        # 将 Gf.Matrix3d 转换为 numpy 数组，然后使用 scipy 转换为四元数
        object_rot_array = np.array([[object_rot_matrix[i][j] for j in range(3)] for i in range(3)])
        gripper_rot_array = np.array([[gripper_rot_matrix[i][j] for j in range(3)] for i in range(3)])
        
        object_quat = R.from_matrix(object_rot_array).as_quat()  # [x, y, z, w]
        gripper_quat = R.from_matrix(gripper_rot_array).as_quat()  # [x, y, z, w]
        
        # 转换为 Gf.Quatd (w, x, y, z) 格式
        self.initial_object_rotation = Gf.Rotation(Gf.Quatd(object_quat[3], object_quat[0], object_quat[1], object_quat[2]))
        self.initial_gripper_rotation = Gf.Rotation(Gf.Quatd(gripper_quat[3], gripper_quat[0], gripper_quat[1], gripper_quat[2]))

    def update_grasped_object_position(self):
        if not self.grasped_object_path or not self.gripper_frame_path:
            return

        
        target_frame_prim = get_prim_at_path(self.gripper_frame_path)
        if not target_frame_prim.IsValid():
            raise ValueError(f"Gripper frame at path {self.gripper_frame_path} is not valid.")

        
        target_world_transform = UsdGeom.Xformable(target_frame_prim).ComputeLocalToWorldTransform(0)
        target_world_position = target_world_transform.ExtractTranslation()
        target_world_rot_matrix = target_world_transform.ExtractRotationMatrix()
        
        # 将 Gf.Matrix3d 转换为 numpy 数组，然后使用 scipy 转换为四元数
        target_rot_array = np.array([[target_world_rot_matrix[i][j] for j in range(3)] for i in range(3)])
        target_quat = R.from_matrix(target_rot_array).as_quat()  # [x, y, z, w]
        
        # 转换为 Gf.Quatd (w, x, y, z) 格式
        target_world_rotation = Gf.Rotation(Gf.Quatd(target_quat[3], target_quat[0], target_quat[1], target_quat[2]))

        
        local_position = self.inverse_transform_matrix.TransformAffine(target_world_position)
        
        object_prim = get_prim_at_path(self.grasped_object_path)
        if not object_prim.IsValid():
            raise ValueError(f"Object at path {self.grasped_object_path} is not valid.")

        if self.position_offest is None:
            self.position_offest = UsdGeom.Xformable(object_prim).GetOrderedXformOps()[0].Get() - local_position 
        
        # 计算对象的新旋转：保持与夹爪的相对旋转关系
        # 旋转差异 = 当前夹爪旋转 * 初始夹爪旋转的逆
        rotation_diff = target_world_rotation * self.initial_gripper_rotation.GetInverse()
        # 新对象旋转 = 旋转差异 * 初始对象旋转
        new_object_rotation = rotation_diff * self.initial_object_rotation
        
        # 更新位置和方向
        xformable = UsdGeom.Xformable(object_prim)
        xform_ops = xformable.GetOrderedXformOps()
        
        # 更新位置
        if xform_ops:
            translate_op = xform_ops[0]
            translate_op.Set(local_position+self.position_offest)
        else:
            xformable.AddTranslateOp().Set(local_position+self.position_offest)
        
        # 更新方向：转换为欧拉角
        euler_angles = new_object_rotation.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
        
        # 检查是否已有旋转操作
        has_rotate_op = False
        for op in xform_ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                op.Set(euler_angles)
                has_rotate_op = True
                break
        
        if not has_rotate_op:
            # 如果没有旋转操作，添加一个
            rotate_op = xformable.AddRotateXYZOp()
            rotate_op.Set(euler_angles)

    def release_object(self):
        self.grasped_object_path = None
        self.gripper_frame_path = None
        self.position_offest = None
        self.initial_object_rotation = None
        self.initial_gripper_rotation = None