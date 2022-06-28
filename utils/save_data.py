# Thanks to the courtesy of:
# https://github.com/carla-simulator/carla/blob/marcel/gbuffer_view/PythonAPI/examples/tutorial_gbuffer.py

# === Save Data ===

exist_idx = []
save_idx = 0

def get_save_idx(frame_idx):

    return 2

    global save_idx
    global exist_idx

    if len(exist_idx) > 0 and frame_idx == exist_idx[-1]:
        return save_idx
    
    exist_idx.append(frame_idx)
    save_idx += 1
    print("exiting with %d" % save_idx)

    return save_idx

def init_save_idx():
    global save_idx
    global exist_idx
    save_idx = 0
    exist_idx = []


def save_function_factory(save_dir):


    def SaveSceneColorTexture(image):
        save_idx = get_save_idx(image.frame)
        image.save_to_disk(f'{save_dir}/SceneColor-%06d.png' % save_idx)

    def SaveSceneDepthTexture(image):
        save_idx = get_save_idx(image.frame)
        image.save_to_disk(f'{save_dir}/SceneDepth-%06d.png' % save_idx)

    def SaveSceneStencilTexture(image):
        save_idx = get_save_idx(image.frame)
        image.save_to_disk(f'{save_dir}/SceneStencil-%06d.png' % save_idx)

    def SaveGBufferATexture(image):
        save_idx = get_save_idx(image.frame)
        image.save_to_disk(f'{save_dir}/GBufferA-%06d.png' % save_idx)

    def SaveGBufferBTexture(image):
        save_idx = get_save_idx(image.frame)
        image.save_to_disk(f'{save_dir}/GBufferB-%06d.png' % save_idx)

    def SaveGBufferCTexture(image):
        save_idx = get_save_idx(image.frame)
        image.save_to_disk(f'{save_dir}/GBufferC-%06d.png' % save_idx)

    def SaveGBufferDTexture(image):
        save_idx = get_save_idx(image.frame)
        image.save_to_disk(f'{save_dir}/GBufferD-%06d.png' % save_idx)

    def SaveGBufferETexture(image):
        save_idx = get_save_idx(image.frame)
        image.save_to_disk(f'{save_dir}/GBufferE-%06d.png' % save_idx)

    def SaveGBufferFTexture(image):
        save_idx = get_save_idx(image.frame)
        image.save_to_disk(f'{save_dir}/GBufferF-%06d.png' % save_idx)

    def SaveVelocityTexture(image):
        save_idx = get_save_idx(image.frame)
        image.save_to_disk(f'{save_dir}/Velocity-%06d.png' % save_idx)

    def SaveSSAOTexture(image):
        save_idx = get_save_idx(image.frame)
        image.save_to_disk(f'{save_dir}/SSAO-%06d.png' % save_idx)

    def SaveCustomDepthTexture(image):
        save_idx = get_save_idx(image.frame)
        image.save_to_disk(f'{save_dir}/CustomDepth-%06d.png' % save_idx)

    def SaveCustomStencilTexture(image):
        save_idx = get_save_idx(image.frame)
        image.save_to_disk(f'{save_dir}/CustomStencil-%06d.png' % save_idx)

    return {
        "SaveSceneColorTexture": SaveSceneColorTexture,
        "SaveSceneDepthTexture": SaveSceneDepthTexture,
        "SaveSceneStencilTexture": SaveSceneStencilTexture,
        "SaveGBufferATexture": SaveGBufferATexture,
        "SaveGBufferBTexture": SaveGBufferBTexture,
        "SaveGBufferCTexture": SaveGBufferCTexture,
        "SaveGBufferDTexture": SaveGBufferDTexture,
        "SaveGBufferETexture": SaveGBufferETexture,
        "SaveGBufferFTexture": SaveGBufferFTexture,
        "SaveVelocityTexture": SaveVelocityTexture,
        "SaveSSAOTexture": SaveSSAOTexture,
        "SaveCustomDepthTexture": SaveCustomDepthTexture,
        "SaveCustomStencilTexture": SaveCustomStencilTexture,

    }