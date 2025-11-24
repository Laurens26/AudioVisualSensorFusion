from gen_model import model
from gen_truth import truth
from gen_meas import meas
from plot_results import plot_results
from run_filter import GLMB, filter
from plot_results import write_results_jsonl
import os

if __name__ == '__main__':
    # Define object tracking model
    model_params = model()
    
    # Define ground truth trajectories
    # A) Use generated ones
    # truth_params = truth(model_params, generate_truth=True)

    # B) Use from groundtruth_sources.jsonl
    # root_dir = "E:/Masterarbeit_Sillekens/Unity Projects/VisualSimulationUnity/Assets/AV_Simulation_Run3"
    root_dir = "E:/Masterarbeit_Sillekens/Unity Projects/VisualSimulationUnity_Experiments/Scenario3"
    truth_params = truth(model_params)
    truth_params.run_jsonl_main_loop(jsonl_filepath = os.path.join(root_dir, 'groundtruth_sources.jsonl'),
                                     dim=2)

    # Define measurements
    # A) Use generated ones
    # meas_params = meas(model_params, truth_params, generate_measurements=True)

    # B) Use audio_localizations.jsonl
    # TODO: 1) Timestamps werden aktuell auf 0.1ms aufgerundet zB 0.1s --> k=10. Da wir teilweise in 100ms mehrere Audio Detektionen haben, wird die vorheriger überschrieben
    #          das führt dazu dass Detektionen verloren gehen. Hier muss eine Lösung für die auflösung der konstanten Zeit k gefunden werden
    # meas_params.run_jsonl_main_loop(jsonl_filepath = os.path.join(root_dir, 'localization/audio_localizations.jsonl'), 
    #                               dim=2, 
    #                               sensor_idx=0)

    # C) Use video_localizations.jsonl
    meas_params = meas(model_params, truth_params)
    meas_params.run_jsonl_main_loop(jsonl_filepath = os.path.join(root_dir, 'localization/video_localizations.jsonl'), 
                                    dim=2, 
                                    sensor_idx=0)

    # D) Use audio_localizations.jsonl and video_localizations.jsonl
    # MAKE SURE TO ALWAYS USE THE MOST RELIABLE SENSOR AS SENSOR 0. SENSOR 0 PARAMS ARE ALSO USED FOR ADAPTH BIRTH (SENSOR 1 PARAMS NOT!)
    # meas_params = meas(model_params, truth_params)
    # meas_params.run_jsonl_main_loop(jsonl_filepath = os.path.join(root_dir, 'localization/video_localizations.jsonl'), 
    #                                 dim=2, 
    #                                 sensor_idx=0)
    # meas_params.run_jsonl_main_loop(jsonl_filepath = os.path.join(root_dir, 'localization/audio_localizations.jsonl'), 
    #                               dim=2, 
    #                               sensor_idx=1)

    
    

    # Define filter
    glmb = GLMB(model_params)
    filter_params = filter(model_params)

    # Run filter to compute state estimations
    est_params = glmb.run(model_params, filter_params, meas_params)
    # glmb.plot_tracks(model_params, filter_params, truth_params, meas_params)
    
    # Plot filter results
    handles = plot_results(model_params, truth_params, meas_params, est_params)

    # Write data to "gt.jsonl" and "tracking.jsonl"
    write_results_jsonl(model_params, truth_params, filename='groundtruth_sources_example.jsonl')
    write_results_jsonl(model_params, est_params, filename='fusion_localizations_example.jsonl')
