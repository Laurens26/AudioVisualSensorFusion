from gen_model import model
from gen_truth import truth
from gen_meas import meas
from plot_results import plot_results
from run_filter import GLMB, filter
import os

if __name__ == '__main__':
    # Define object tracking model
    model_params = model()
    
    # Define ground truth trajectories
    # A) Use generated ones
    truth_params = truth(model_params)

    # B) Use from groundtruth_sources.jsonl
    root_dir = "E:/Masterarbeit_Sillekens/Unity Projects/VisualSimulationUnity/Assets/AV_Simulation_Run1"
    truth_params = truth(model_params, jsonl_filepath = os.path.join(root_dir, 'groundtruth_sources.jsonl'))

    # Define measurements
    # A) Use generated ones
    # meas_params = meas(model_params, truth_params)

    # B) Use audio_localizations.jsonl
    # TODO: 1) Timestamps werden aktuell auf 0.1ms aufgerundet zB 0.1s --> k=10. Da wir teilweise in 100ms Audio Detektionen haben, wird die vorheriger überschrieben
    #          das führt dazu dass Detektionen verloren gehen. Hier muss eine Lösung für die auflösung der konstanten Zeit k gefunden werden
    #       2) Die Detektionsrate in der audio_localization.jsonl ist sehr gering (P_D unter 0.5) was dazuführt, dass führt etliche Zeitstempel k keine Messungen 
    #          vorhanden sind. Das führt dazu, dass GLMB nicht richtig tracken kann. Es gilt herauszufinden, ob GLMB nicht trotzdem mit wenigen Detektions arbeiten kann 
    # meas_params = meas(model_params, truth_params, jsonl_filepath = os.path.join(root_dir, 'localization/audio_localizations.jsonl'))

    # C) Use video_localizations.jsonl
    # TODO: Probably same issue as with audio_localizations.jsonl
    meas_params = meas(model_params, truth_params, jsonl_filepath = os.path.join(root_dir, 'localization/video_localizations.jsonl'))

    # Define filter
    glmb = GLMB(model_params)
    filter_params = filter(model_params)

    # Run filter to compute state estimations
    est_params = glmb.run(model_params, filter_params, meas_params)
    # glmb.plot_tracks(model_params, filter_params, truth_params, meas_params)
    
    # Plot filter results
    handles = plot_results(model_params, truth_params, meas_params, est_params)
