# Thanks to the courtesy of:
# https://github.com/carla-simulator/carla/blob/marcel/gbuffer_view/PythonAPI/examples/tutorial_gbuffer.py

# === Save Data ===

# exist_idx = []
# save_idx = 0

# def get_save_idx(frame_idx):

#     return frame_idx

#     global save_idx
#     global exist_idx

#     if len(exist_idx) > 0 and frame_idx == exist_idx[-1]:
#         return save_idx
    
#     exist_idx.append(frame_idx)
#     save_idx += 1
#     print("exiting with %d" % image.frame)

#     return save_idx

# def init_save_idx():
#     global save_idx
#     global exist_idx
#     save_idx = 0
#     exist_idx = []


def save_function_factory(save_dir):


    def SaveSceneColorTexture(image):
        image.save_to_disk(f'{save_dir}/SceneColor-%06d.png' % image.frame)

    def SaveSceneDepthTexture(image):
        image.save_to_disk(f'{save_dir}/SceneDepth-%06d.png' % image.frame)

    def SaveSceneStencilTexture(image):
        image.save_to_disk(f'{save_dir}/SceneStencil-%06d.png' % image.frame)

    def SaveGBufferATexture(image):
        image.save_to_disk(f'{save_dir}/GBufferA-%06d.png' % image.frame)

    def SaveGBufferBTexture(image):
        image.save_to_disk(f'{save_dir}/GBufferB-%06d.png' % image.frame)

    def SaveGBufferCTexture(image):
        image.save_to_disk(f'{save_dir}/GBufferC-%06d.png' % image.frame)

    def SaveGBufferDTexture(image):
        image.save_to_disk(f'{save_dir}/GBufferD-%06d.png' % image.frame)

    def SaveGBufferETexture(image):
        image.save_to_disk(f'{save_dir}/GBufferE-%06d.png' % image.frame)

    def SaveGBufferFTexture(image):
        image.save_to_disk(f'{save_dir}/GBufferF-%06d.png' % image.frame)

    def SaveVelocityTexture(image):
        image.save_to_disk(f'{save_dir}/Velocity-%06d.png' % image.frame)

    def SaveSSAOTexture(image):
        image.save_to_disk(f'{save_dir}/SSAO-%06d.png' % image.frame)

    def SaveCustomDepthTexture(image):
        image.save_to_disk(f'{save_dir}/CustomDepth-%06d.png' % image.frame)

    def SaveCustomStencilTexture(image):
        image.save_to_disk(f'{save_dir}/CustomStencil-%06d.png' % image.frame)

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