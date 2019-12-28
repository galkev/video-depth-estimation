import bpy

bl_info = {
    "name": "Save Camera Pos",
    "category": "Object",
}


class SaveCamPos(bpy.types.Operator):
    bl_idname = "object.save_cam_pos"  # Unique identifier for buttons and menu items to reference.
    bl_label = bl_info["name"]  # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}  # Enable undo for the operator.

    def execute(self, context):  # execute() is called when running the operator.
        # The original script
        cam = bpy.data.objects["Camera"]

        o = bpy.data.objects.new("campos", None)
        o.location = cam.location
        o.rotation_euler = cam.rotation_euler
        bpy.context.scene.objects.link(o)

        return {'FINISHED'}  # Lets Blender know the operator finished successfully.


def register():
    bpy.utils.register_class(SaveCamPos)


def unregister():
    bpy.utils.unregister_class(SaveCamPos)


# This allows you to run the script directly from Blender's Text editor
# to test the add-on without having to install it.
if __name__ == "__main__":
    register()
