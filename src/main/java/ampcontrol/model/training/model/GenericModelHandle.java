package ampcontrol.model.training.model;

import ampcontrol.model.training.data.iterators.MiniEpochDataSetIterator;
import ampcontrol.model.training.listen.NanScoreWatcher;
import ampcontrol.model.training.model.validation.Validation;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Optional;

/**
 * Handles fitting and evaluation of models encapsulated by a {@link ModelAdapter}. Manages {@link MiniEpochDataSetIterator}s
 * for training and evaluation and suspends training in case score or activation of the model is NaN.
 *
 * @author Christian Skärby
 */
public class GenericModelHandle implements ModelHandle {

    private static final Logger log = LoggerFactory.getLogger(GenericModelHandle.class);

    private static int nanTimeOutTime = 200;

    private final MiniEpochDataSetIterator trainingIter;
    private final MiniEpochDataSetIterator evalIter;
    private final ModelAdapter model;
    private final String name;
    private final Collection<Validation<? extends IEvaluation<?>>> validations = new ArrayList<>();

    private int nanTimeOutTimer = nanTimeOutTime;

    /**
     * Constructor
     *
     * @param trainingIter Provides training samples.
     * @param evalIter     Provides evaluation samples.
     * @param model        Model to fit/evaluate
     * @param name         Name of the model
     */
    public GenericModelHandle(MiniEpochDataSetIterator trainingIter, MiniEpochDataSetIterator evalIter, ModelAdapter model, String name) {
        this.trainingIter = trainingIter;
        this.evalIter = evalIter;
        this.model = model;
        this.name = name;
        model.asModel().addListeners(new NanScoreWatcher(() -> nanTimeOutTimer = 0));
    }


    @Override
    public void registerValidation(Validation.Factory<? extends IEvaluation<?>> validationFactory) {
        validations.add(validationFactory.create(evalIter.getLabels()));
    }

    @Override
    public void addListener(TrainingListener listener) {
        model.asModel().addListeners(listener);
    }

    @Override
    public void saveModel(String fileName) throws IOException {
        log.info("Saving model: " + name() + " as " + fileName);
        model.saveModel(fileName);
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public void fit() {
        if (nanTimeOutTimer < nanTimeOutTime) {
            //TODO: reload model when timer is up
            // nanTimeOutTimer++;
            log.warn("Model " + name() + " broken. Skipping...");
            return;
        }

        trainingIter.restartMiniEpoch();

        model.fit(trainingIter);
    }

    @Override
    public void eval() {
        if (nanTimeOutTimer < nanTimeOutTime) {
            return;
        }

        evalIter.restartMiniEpoch();

        final IEvaluation<?>[] evalArr = validations.stream()
                .map(Validation::get)
                .filter(Optional::isPresent)
                .map(Optional::get)
                .toArray(IEvaluation<?>[]::new);

        if (evalArr.length > 0) {
            model.eval(evalIter, evalArr);
        }
        validations.forEach(Validation::notifyComplete);
    }

    @Override
    public void resetTraining() {
        trainingIter.reset();
    }
}
