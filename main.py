import sys

from airline_satisfaction.preprocessing import main as preprocess_main
from airline_satisfaction.cross_validation import main as cv_main


def main():
    args = sys.argv[1:]
    if args and args[0] == 'cv':
        cv_main(args[1:])
    elif args and args[0] == 'baseline':
        from airline_satisfaction.baseline import main as baseline_main
        baseline_main(args[1:])
    elif args and args[0] == 'evaluate':
        from airline_satisfaction.evaluation import main as evaluate_main
        evaluate_main(args[1:])
    elif args and args[0] == 'features':
        from airline_satisfaction.feature_experiments import main as features_main
        features_main(args[1:])
    elif args and args[0] == 'tune':
        from airline_satisfaction.tuning import main as tune_main
        tune_main(args[1:])
    elif args and args[0] == 'ensemble':
        from airline_satisfaction.ensemble import main as ensemble_main
        ensemble_main(args[1:])
    elif args and args[0] == 'targetstats':
        from airline_satisfaction.target_statistics import main as targetstats_main
        targetstats_main(args[1:])
    elif args and args[0] == 'explore':
        from airline_satisfaction.lgbm_exploration import main as explore_main
        explore_main(args[1:])
    elif args and args[0] == 'final':
        from airline_satisfaction.final_model import main as final_main
        final_main(args[1:])
    elif args and args[0] == 'multimodel':
        from airline_satisfaction.multimodel import main as multimodel_main
        multimodel_main(args[1:])
    else:
        preprocess_main(args[1:] if args and args[0] == 'preprocess' else args)


if __name__ == "__main__":
    main()
