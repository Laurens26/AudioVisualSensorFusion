from gen_model import model
from gen_truth import truth
from gen_meas import meas
from plot_results import plot_results
from run_filter import GLMB, filter
from plot_results import write_results_jsonl
import os

if __name__ == '__main__':
    # Define object tracking model
    model_params = model() # Linear kalman filter with constant velocity model
    
    # Define or read groundtruth tracks
    # A) Use generated groundtruth tracks
    # truth_params = truth(model_params, generate_truth=True)

    # B) Use groundtruth tracks from JSONL: groundtruth_sources.jsonl
    root_dir = "E:/Masterarbeit_Sillekens/Unity Projects/VisualSimulationUnity_Experiments/Scenario1"
    truth_params = truth(model_params)
    truth_params.run_jsonl_main_loop(jsonl_filepath = os.path.join(root_dir, 'groundtruth_sources.jsonl'),
                                     dim=2)

    # Define or read measurements (video/audio detections)
    # A) Use generated ones
    # meas_params = meas(model_params, truth_params, generate_measurements=True)

    # B) Use audio detections only: audio_localizations.jsonl
    # meas_params = meas(model_params, truth_params)
    # meas_params.run_jsonl_main_loop(jsonl_filepath = os.path.join(root_dir, 'localization/audio_localizations.jsonl'), 
    #                               dim=2, 
    #                               sensor_idx=0)     
    # filename = f"audio_tracking_{os.path.basename(root_dir)}.jsonl"

    # C) Use video detections only: video_localizations.jsonl
    # meas_params = meas(model_params, truth_params)
    # meas_params.run_jsonl_main_loop(jsonl_filepath = os.path.join(root_dir, 'localization/video_localizations.jsonl'), 
    #                                 dim=2, 
    #                                 sensor_idx=0)
    # filename = f"video_tracking_{os.path.basename(root_dir)}.jsonl"

    # D) Use audio and video detections: audio_localizations.jsonl and video_localizations.jsonl
    # MAKE SURE TO ALWAYS USE THE MOST RELIABLE SENSOR AS SENSOR 0. SENSOR 0 PARAMS ARE ALSO USED FOR ADAPTH BIRTH (SENSOR 1 PARAMS NOT!)
    meas_params = meas(model_params, truth_params)
    meas_params.run_jsonl_main_loop(jsonl_filepath = os.path.join(root_dir, 'localization/video_localizations.jsonl'), 
                                    dim=2, 
                                    sensor_idx=0)
    meas_params.run_jsonl_main_loop(jsonl_filepath = os.path.join(root_dir, 'localization/audio_localizations.jsonl'), 
                                  dim=2, 
                                  sensor_idx=1)
    filename = f"audio_video_tracking_{os.path.basename(root_dir)}.jsonl"

    # Define MS-GLMB filter
    glmb = GLMB(model_params)
    filter_params = filter(model_params)

    # Run filter to compute state estimations
    est_params = glmb.run(model_params, filter_params, meas_params)
    # glmb.plot_tracks(model_params, filter_params, truth_params, meas_params)
    
    # Plot filter results
    handles = plot_results(model_params, truth_params, meas_params, est_params, title=os.path.basename(root_dir), filename=os.path.splitext(filename)[0]) # title e.g. Scenario1

    # Write predicted tracks as JSONL: e.g. audio_video_tracking_Scenario1.jsonl
    write_results_jsonl(model_params, est_params, filename) 

