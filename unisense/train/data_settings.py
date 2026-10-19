import os


data_root = './data'
datasets_root = './dataset'

CONVERSATION_DATA = dict(
    # Preliminary speech alignment and image-speech-text instruction tuning.
    unisense_pretrain=dict(
        image_folder=os.environ.get('PRETRAIN_SPEECH_FOLDER', os.path.join(data_root, 'Unisense_Pretrain')),
        speech_folder=os.environ.get('PRETRAIN_SPEECH_FOLDER', os.path.join(data_root, 'Unisense_Pretrain')),
        data_path=os.environ.get('PRETRAIN_DATA_PATH', os.path.join(data_root, 'Unisense_Pretrain/unisense_pretrain.json')),
        frequency=1,
    ),
    unisense_sft_1m_fake_multi_3_rounds=dict(
        image_folder=os.environ.get('MULTIMODAL_IMAGE_FOLDER', os.path.join(data_root, 'Unisense_SFT/multi_modality_image')),
        speech_folder=os.environ.get('MULTIMODAL_SPEECH_FOLDER', os.path.join(data_root, 'Unisense_SFT/multi_modality_speech')),
        data_path=os.environ.get('MULTIMODAL_SFT_DATA_PATH', os.path.join(data_root, 'fake_multi_conversations/unisense_multimodal_fake_multi_3_rounds.json')),
        frequency=1,
    ),
    finevision_speech_14k_fake_multi_3_rounds=dict(
        image_folder=os.environ.get('FINEVISION_FOLDER', os.path.join(data_root, 'FineVision')),
        speech_folder=os.environ.get('FINEVISION_FOLDER', os.path.join(data_root, 'FineVision')),
        data_path=os.environ.get('FINEVISION_SPEECH_DATA_PATH', os.path.join(data_root, 'fake_multi_conversations/finevision_speech_14k_fake_multi_3_rounds.jsonl')),
        frequency=1,
    ),
    finevision_349k_fake_multi_3_rounds=dict(
        image_folder=os.environ.get('FINEVISION_FOLDER', os.path.join(data_root, 'FineVision')),
        speech_folder=os.environ.get('FINEVISION_FOLDER', os.path.join(data_root, 'FineVision')),
        data_path=os.environ.get('FINEVISION_DATA_PATH', os.path.join(data_root, 'fake_multi_conversations/finevision_349k_fake_multi_3_rounds.jsonl')),
        frequency=1,
    ),
    ################################### Depth ##################################
    spatialqa_4k_obj_caption_fake_multi_3_rounds=dict(
        image_folder=os.path.join(datasets_root, 'Depth/SpatialQA'),
        depth_folder=os.path.join(datasets_root, 'Depth/SpatialQA'),
        data_path=os.path.join(data_root, 'fake_multi_conversations/spatialqa_4k_obj_caption_fake_multi_3_rounds.jsonl'),
        frequency=1,
    ),
    spatialqa_10k_obj_caption_fake_multi_3_rounds=dict(
        image_folder=os.path.join(datasets_root, 'Depth/SpatialQA'),
        speech_folder=os.path.join(datasets_root, 'Depth/SpatialQA'),
        depth_folder=os.path.join(datasets_root, 'Depth/SpatialQA'),
        data_path=os.path.join(data_root, 'fake_multi_conversations/spatialqa_10k_obj_caption_fake_multi_3_rounds.jsonl'),
        frequency=1,
    ),
    spar_10k_obj_caption_fake_multi_3_rounds=dict(
        image_folder=os.path.join(datasets_root, 'Depth/spar/color_image'),
        speech_folder=os.path.join(datasets_root, 'Depth/spar'),
        depth_folder=os.path.join(datasets_root, 'Depth/spar/depth_image'),
        data_path=os.path.join(data_root, 'fake_multi_conversations/spar_10k_obj_caption_fake_multi_3_rounds.jsonl'),
        frequency=1,
    ),
    cavqa_train_11k_obj_caption_fake_multi_3_rounds=dict(
        image_folder=os.path.join(datasets_root, 'Depth/CAVQA/cavqa_val_filtered'),
        speech_folder=os.path.join(datasets_root, 'Depth/CAVQA/cavqa_val_filtered'),
        depth_folder=os.path.join(datasets_root, 'Depth/CAVQA/cavqa_val_filtered'),
        data_path=os.path.join(data_root, 'fake_multi_conversations/cavqa_sft_train_obj_caption_fake_multi_3_rounds.jsonl'),
        frequency=1,
    ),

    ################################### Pointcloud #############################
    clevr3d_qa_train_38k=dict(
        pointcloud_folder=os.path.join(datasets_root, 'pointcloud/SceneVerse_preprocessed/pointcloud'),
        data_path=os.path.join(data_root, 'pointcloud/clevr3D_caption_train_38k.jsonl'),
        frequency=1,
    ),
    scanqa_train_25k=dict(
        pointcloud_folder=os.path.join(datasets_root, 'pointcloud/SceneVerse_preprocessed/pointcloud'),
        data_path=os.path.join(data_root, 'pointcloud/scanQA_caption_train_25k.jsonl'),
        frequency=1,
    ),
    sceneverse_full_6k=dict(
        pointcloud_folder=os.path.join(datasets_root, 'pointcloud/SceneVerse_preprocessed/pointcloud'),
        data_path=os.path.join(data_root, 'pointcloud/sceneverse_full_caption_data_cap1_6k.jsonl'),
        frequency=1,
    ),
    sceneverse_train_1k=dict(
        pointcloud_folder=os.path.join(datasets_root, 'pointcloud/SceneVerse_preprocessed/pointcloud'),
        data_path=os.path.join(data_root, 'pointcloud/sceneverse_train_caption_data_cap1_1k.jsonl'),
        frequency=1,
    ),
    scanqa_train_image_caption_25k=dict(
        image_folder=os.path.join(datasets_root, 'pointcloud/MV-ScanQA/frames_square'),
        pointcloud_folder=os.path.join(datasets_root, 'pointcloud/SceneVerse_preprocessed/pointcloud'),
        data_path=os.path.join(data_root, 'pointcloud/scanqa_with_image_train.jsonl'),
        frequency=1,
    ),
    scanqa_train_multiview_caption_7k=dict(
        image_folder=os.path.join(datasets_root, 'pointcloud/MV-ScanQA/frames_square'),
        pointcloud_folder=os.path.join(datasets_root, 'pointcloud/SceneVerse_preprocessed/pointcloud'),
        data_path=os.path.join(data_root, 'pointcloud/mvscanqa_train_7k.jsonl'),
        frequency=1,
    ),

    ################################### IMU ####################################
    llasa_train_v2=dict(
        imu_folder=os.path.join(datasets_root, 'IMU'),
        data_path=os.path.join(data_root, 'IMU/llasa_training_imu_v2_train.jsonl'),
        frequency=1,
    ),
    llasa_train_v2_5qa=dict(
        imu_folder=os.path.join(datasets_root, 'IMU'),
        data_path=os.path.join(data_root, 'IMU/llasa_training_imu_v2_5qa.jsonl'),
        frequency=1,
    ),
    llasa_classification_23k=dict(
        imu_folder=os.path.join(datasets_root, 'IMU'),
        data_path=os.path.join(data_root, 'IMU/llasa_train_imu_v1_classification_open_23k.jsonl'),
        frequency=2,
    ),
    # Unweighted variant used by the 9B C2 recipe.
    llasa_classification_23k_x1=dict(
        imu_folder=os.path.join(datasets_root, 'IMU'),
        data_path=os.path.join(data_root, 'IMU/llasa_train_imu_v1_classification_open_23k.jsonl'),
        frequency=1,
    ),

    ################################### Tactile ################################
    tactile_octopi_50k=dict(
        tactile_folder=os.path.join(datasets_root, 'Tactile/octopi/tactile_images'),
        data_path=os.path.join(data_root, 'Tactile/train_qa_tactile_octopi_50k.jsonl'),
        frequency=1,
    ),
    tactile_VTV_50k=dict(
        tactile_folder=os.path.join(datasets_root, 'Tactile/VTV-LLM/tactile_images'),
        data_path=os.path.join(data_root, 'Tactile/train_qa_tactile_VTV_50k.jsonl'),
        frequency=1,
    ),

    ################################### Thermal ################################
    thermal_IRTD_qa_v1_80k=dict(
        image_folder=os.path.join(datasets_root, 'Thermal'),
        thermal_folder=os.path.join(datasets_root, 'Thermal'),
        data_path=os.path.join(data_root, 'Thermal/thermal_IRTD_qa_train_v1_80k.jsonl'),
        frequency=1,
    ),
    thermal_IRTD_caption=dict(
        image_folder=os.path.join(datasets_root, 'Thermal'),
        thermal_folder=os.path.join(datasets_root, 'Thermal'),
        data_path=os.path.join(data_root, 'Thermal/thermal_IRTD_caption_train.jsonl'),
        frequency=1,
    ),
    thermal_IRTD_task_train=dict(
        image_folder=os.path.join(datasets_root, 'Thermal'),
        thermal_folder=os.path.join(datasets_root, 'Thermal'),
        data_path=os.path.join(data_root, 'Thermal/thermal_IRTD_task_train.jsonl'),
        frequency=1,
    ),
    thermal_IRTD_task_train_no_cap=dict(
        image_folder=os.path.join(datasets_root, 'Thermal'),
        thermal_folder=os.path.join(datasets_root, 'Thermal'),
        data_path=os.path.join(data_root, 'Thermal/thermal_IRTD_task_train_no_cap.jsonl'),
        frequency=1,
    ),
    thermal_MSPR_train_no_cap=dict(
        thermal_folder=os.path.join(datasets_root, 'Thermal'),
        data_path=os.path.join(data_root, 'Thermal/thermal_MSPR_train_no_cap.jsonl'),
        frequency=1,
    ),
    thermal_infrared_llava_train_no_cap=dict(
        thermal_folder=os.path.join(datasets_root, 'Thermal/infrared_llava/benchmark/infrared'),
        data_path=os.path.join(data_root, 'Thermal/thermal_infrared_llava_train_no_cap.jsonl'),
        frequency=1,
    ),

    ################################### Radar ##################################
    radar_holollm_mmfivqa=dict(
        radar_folder=os.path.join(datasets_root, 'Radar/MMFi_dataset/filtered_mmwave'),
        data_path=os.path.join(data_root, 'Radar/holollm_mmfivqa_mmwave_v5.jsonl'),
        frequency=1,
    ),
    radar_holollm_mmfi_cap=dict(
        radar_folder=os.path.join(datasets_root, 'Radar/MMFi_dataset/filtered_mmwave'),
        data_path=os.path.join(data_root, 'Radar/holollm_mmfi_cap_v5.jsonl'),
        frequency=1,
    ),
    radar_mmfi_actionreg=dict(
        radar_folder=os.path.join(datasets_root, 'Radar/MMFi_dataset/filtered_mmwave'),
        data_path=os.path.join(data_root, 'Radar/holollm_mmfivqa_mmwave_eval_fixed_v6.jsonl'),
        frequency=1,
    ),
    radar_mmfi_direction_mcq=dict(
        radar_folder=os.path.join(datasets_root, 'Radar/MMFi_dataset/filtered_mmwave'),
        data_path=os.path.join(data_root, 'Radar/mmfi_direction_mcq_v6.jsonl'),
        frequency=1,
    ),
    radar_mmfi_confusion_mcq=dict(
        radar_folder=os.path.join(datasets_root, 'Radar/MMFi_dataset/filtered_mmwave'),
        data_path=os.path.join(data_root, 'Radar/mmfi_confusion_mcq_v5.jsonl'),
        frequency=1,
    ),
    radar_mmfi_general_mcq=dict(
        radar_folder=os.path.join(datasets_root, 'Radar/MMFi_dataset/filtered_mmwave'),
        data_path=os.path.join(data_root, 'Radar/mmfi_general_mcq_v5.jsonl'),
        frequency=1,
    ),

    ################################### Smell (raw) ############################
    smell_caption_sft_train_raw=dict(
        smell_folder=os.path.join(datasets_root, 'smell/smell_data_raw'),
        data_path=os.path.join(data_root, 'Smell/smell_caption_sft_train.jsonl'),
        frequency=1,
    ),
    smell_classification_train_raw=dict(
        smell_folder=os.path.join(datasets_root, 'smell/smell_data_raw'),
        data_path=os.path.join(data_root, 'Smell/smell_classification_train.jsonl'),
        frequency=1,
    ),
    smell_mcq_classification_train_raw=dict(
        smell_folder=os.path.join(datasets_root, 'smell/smell_data_raw'),
        data_path=os.path.join(data_root, 'Smell/smell_mcq_classification_train.jsonl'),
        frequency=1,
    ),
    smell_mcq_category_train_raw=dict(
        smell_folder=os.path.join(datasets_root, 'smell/smell_data_raw'),
        data_path=os.path.join(data_root, 'Smell/smell_mcq_category_train.jsonl'),
        frequency=1,
    ),
    smell_judgment_binary_train_raw=dict(
        smell_folder=os.path.join(datasets_root, 'smell/smell_data_raw'),
        data_path=os.path.join(data_root, 'Smell/smell_judgment_binary_train.jsonl'),
        frequency=1,
    ),
    smell_numeric_qa_train_raw=dict(
        smell_folder=os.path.join(datasets_root, 'smell/smell_data_raw'),
        data_path=os.path.join(data_root, 'Smell/smell_numeric_qa_train.jsonl'),
        frequency=1,
    ),

    ################################### Motion #################################
    quants_binary_train=dict(
        motion_folder=os.path.join(datasets_root, 'Motion/motion_data'),
        data_path=os.path.join(data_root, 'Motion/quants_binary_train.jsonl'),
        frequency=1,
    ),
    quants_multi_train=dict(
        motion_folder=os.path.join(datasets_root, 'Motion/motion_data'),
        data_path=os.path.join(data_root, 'Motion/quants_multi_train.jsonl'),
        frequency=1,
    ),
    quants_open_train=dict(
        motion_folder=os.path.join(datasets_root, 'Motion/motion_data'),
        data_path=os.path.join(data_root, 'Motion/quants_open_train.jsonl'),
        frequency=1,
    ),
)
