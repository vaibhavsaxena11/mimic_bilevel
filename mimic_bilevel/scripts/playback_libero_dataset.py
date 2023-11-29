"""
A convenience script to playback random demonstrations from
a set of demonstrations stored in a hdf5 file.

Arguments:
    --folder (str): Path to demonstrations
    --use-actions (optional): If this flag is provided, the actions are played back
        through the MuJoCo simulator, instead of loading the simulator states
        one by one.
    --visualize-gripper (optional): If set, will visualize the gripper site

Example:
    $ python scripts/playback_libero_dataset.py \
        --data-path /Users/vaibhav/work/robomimicV2/LIBERO/datasets/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5 \
        --use-actions \
        --videos-save-folder /Users/vaibhav/work/imitate-behavior/mimic_bilevel/mimic_bilevel/playback_libero_dataset/playback_demo/
"""

import argparse
import json
import os
import random
import pathlib
import imageio

import h5py
import numpy as np

import robosuite
import libero

import robomimic.utils.env_utils as EnvUtils
import robomimic.envs.env_base as EB
import robomimic.utils.obs_utils as ObsUtils

from libero.libero.envs.problems.libero_kitchen_tabletop_manipulation import Libero_Kitchen_Tabletop_Manipulation

import robosuite
import xml.etree.ElementTree as ET
from robosuite.utils.mjcf_utils import find_elements
def postprocess_model_xml(xml_str, cameras_dict={}, verbose=False):
    """
    This function postprocesses the model.xml collected from a MuJoCo demonstration
    in order to make sure that the STL files can be found.

    Args:
        xml_str (str): Mujoco sim demonstration XML file as string

    Returns:
        str: Post-processed xml file as string
    """

    path = os.path.split(robosuite.__file__)[0]
    path_split = path.split("/")

    # replace mesh and texture file paths
    tree = ET.fromstring(xml_str)
    root = tree
    asset = root.find("asset")
    meshes = asset.findall("mesh")
    textures = asset.findall("texture")
    all_elements = meshes + textures

    # also replace paths for libero
    libero_path = libero.__path__[0] + "/libero"
    libero_path_split = libero_path.split("/")

    print("gathered all elems")

    for elem in all_elements:
        old_path = elem.get("file")
        if old_path is None:
            continue
        old_path_split = old_path.split("/")
        if "robosuite" in old_path_split:
            ind = max(
                loc for loc, val in enumerate(old_path_split) if val == "robosuite"
            )  # last occurrence index
            new_path_split = path_split + old_path_split[ind + 1 :]
            new_path = "/".join(new_path_split)
            elem.set("file", new_path)
            if verbose:
                print(f"edited robosuite asset path to: {new_path}")
        # elif "libero" in old_path_split and demo_generation:
        elif any(['libero' in pth for pth in old_path_split]):
            ind = max(
                # loc for loc, val in enumerate(old_path_split) if val == "libero"
                loc for loc, val in enumerate(old_path_split) if "libero" in val
            )  # last occurrence index
            # new_path_split = libero_path_split + old_path_split[ind + 1 :]
            new_path_split = libero_path_split + old_path_split[ind + 2 :]
            new_path = "/".join(new_path_split)
            elem.set("file", new_path)
            if verbose:
                print(f"edited libero asset path to: {new_path}")
        else:
            continue

    # cameras = root.find("worldbody").findall("camera")
    cameras = find_elements(root=tree, tags="camera", return_first=False)
    for camera in cameras:
        camera_name = camera.get("name")
        if camera_name in cameras_dict:
            camera.set("name", camera_name)
            camera.set("pos", cameras_dict[camera_name]["pos"])
            camera.set("quat", cameras_dict[camera_name]["quat"])
            camera.set("mode", "fixed")

    return ET.tostring(root, encoding="utf8").decode("utf8")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to your demonstration folder that contains the demo.hdf5 file, e.g.: "
        "'path_to_assets_dir/demonstrations/YOUR_DEMONSTRATION'",
    ),
    parser.add_argument(
        "--use-actions",
        action="store_true",
    )
    parser.add_argument(
        "--videos-save-folder",
        type=str,
        default=None,
        help="folder to store videos"
    )
    parser.add_argument(
        "--video-skip",
        type=int,
        default=5,
        help="num frames to skip when storing video"
    )
    args = parser.parse_args()

    if args.videos_save_folder is not None:
        os.makedirs(args.videos_save_folder, exist_ok=True)

    hdf5_path = args.data_path
    f = h5py.File(hdf5_path, "r")
    env_name = f["data"].attrs["env_name"]
    env_args = json.loads(f["data"].attrs["env_args"])
    env_kwargs = env_args["env_kwargs"]
    env_kwargs["bddl_file_name"] = str(pathlib.Path(libero.__path__[0]).parent / f["data"].attrs["bddl_file_name"])

    # env_kwargs["render"] = True
    env_kwargs["use_camera_obs"] = True

    
    # env = robosuite.make(
    #     env_name=env_name,
    #     # **env_info,
    #     # has_renderer=True,
    #     # has_offscreen_renderer=False,
    #     # render=True,
    #     # ignore_done=True,
    #     # use_camera_obs=False,
    #     # reward_shaping=True,
    #     # control_freq=20,
    #     **env_kwargs
    # )

    env = EnvUtils.create_env(
        env_type=EB.EnvType.ROBOSUITE_TYPE,
        env_name=env_name,
        render=True,
        **env_kwargs
    )

    dummy_spec = dict(
        obs=dict(
                low_dim=["robot0_eef_pos"],
                rgb=["agentview_image"],
            ),
    )
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

    # list of all demonstrations episodes
    demos = list(f["data"].keys())

    while True:
        print("Playing back random episode... (press ESC to quit)")

        # select an episode randomly
        ep = random.choice(demos)

        # maybe create video writer
        video_count = 0  # video frame counter
        video_writer = None
        if args.videos_save_folder is not None:
            video_writer = imageio.get_writer(os.path.join(args.videos_save_folder, f"playback_{ep}.mp4"), fps=20)

        # read the model xml, using the metadata stored in the attribute for this episode
        model_xml = f["data/{}".format(ep)].attrs["model_file"]

        # env.reset()
        # # xml = env.edit_model_xml(model_xml)
        # # env.reset_from_xml_string(xml)
        # env.env.sim.reset()
        # env.viewer.set_camera(0)

        model_xml = postprocess_model_xml(model_xml, {})
        env.env.reset_from_xml_string(model_xml)
        # env.reset_to({"model": model_xml})

        # load the flattened mujoco states
        states = f["data/{}/states".format(ep)][()]

        if args.use_actions:

            # load the initial state
            # import pdb; pdb.set_trace()
            env.env.sim.set_state_from_flattened(states[0])
            env.env.sim.forward()

            # load the actions and play them back open-loop
            actions = np.array(f["data/{}/actions".format(ep)][()])
            num_actions = actions.shape[0]

            for j, action in enumerate(actions):
                env.step(action)
                env.render()

                # save video
                if video_writer is not None:
                    if video_count % args.video_skip == 0:
                        video_img = []
                        for cam_name in ["agentview"]:
                            video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                        video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                        # import pdb; pdb.set_trace()
                        video_writer.append_data(video_img)
                    video_count += 1

                if j < num_actions - 1:
                    # ensure that the actions deterministically lead to the same recorded states
                    state_playback = env.env.sim.get_state().flatten()
                    if not np.all(np.equal(states[j + 1], state_playback)):
                        err = np.linalg.norm(states[j + 1] - state_playback)
                        print(f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}")

        else:
            # force the sequence of internal mujoco states one by one
            for state in states:
                env.env.sim.set_state_from_flattened(state)
                env.env.sim.forward()
                env.render()

    f.close()
