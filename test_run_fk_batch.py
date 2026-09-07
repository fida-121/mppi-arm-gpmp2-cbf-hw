import mujoco
model = mujoco.MjModel.from_xml_path("assets/panda.xml")
for i in range(1, 8):
    j = model.joint(f"joint{i}")
    b = model.body(f"link{i}")
    print(f"joint{i}: axis={j.axis}, pos={j.pos}, body_pos={model.body_pos[b.id]}, body_quat={model.body_quat[b.id]}")
