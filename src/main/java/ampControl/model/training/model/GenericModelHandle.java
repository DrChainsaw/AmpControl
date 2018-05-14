package ampControl.model.training.model;

import ampControl.model.training.data.iterators.CachingDataSetIterator;
import ampControl.model.training.listen.NanScoreWatcher;
import ampControl.model.training.listen.TrainEvaluator;
import ampControl.model.training.model.validation.Validation;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Optional;
import java.util.function.BiConsumer;

/**
 * Handles fitting and evaluation of models encapsulated by a {@link ModelAdapter}. Manages {@link CachingDataSetIterator
 * CachingDataSetIterators} for training and evaluation and suspends training in case score or activation of the model
 * is NaN.
 *
 * @author Christian Skärby
 */
// TODO: Clean up if possible: getNrof***, getBestEvalScore
public class GenericModelHandle implements ModelHandle {

    private static final Logger log = LoggerFactory.getLogger(GenericModelHandle.class);

    private static int nanTimeOutTime = 200;

    private final CachingDataSetIterator trainingIter;
    private final CachingDataSetIterator evalIter;
    private final ModelAdapter model;
    private final String name;
    private Optional<TrainEvaluator> trainEvaluatorListener = Optional.empty();
    private final Collection<Validation<? extends IEvaluation>> validations = new ArrayList<>();

    private double bestEvalScore;
    private int nanTimeOutTimer = nanTimeOutTime;

    /**
     * Constructor
     *
     * @param trainingIter  Provides training samples.
     * @param evalIter      Provides evaluation samples.
     * @param model         Model to fit/evaluate
     * @param name          Name of the model
     * @param bestEvalScore Initial best evaluation metric found so far.
     */
    public GenericModelHandle(CachingDataSetIterator trainingIter, CachingDataSetIterator evalIter, ModelAdapter model, String name,
                              double bestEvalScore) {
        this.trainingIter = trainingIter;
        this.evalIter = evalIter;
        this.model = model;
        this.name = name;
        this.bestEvalScore = bestEvalScore;

        model.asModel().addListeners(new NanScoreWatcher(() -> nanTimeOutTimer = 0));
    }

    @Override
    public Model getModel() {
        return model.asModel();
    }

    @Override
    public int getNrofBatchesForTraining() {
        return trainingIter.getNrofItersToCache();
    }

    @Override
    public int getNrofTrainingExamplesPerBatch() {
        return trainingIter.numExamples();
    }

    @Override
    public int getNrofEvalExamples() {
        return evalIter.numExamples();
    }

    @Override
    public void createTrainingEvalListener(BiConsumer<Integer, Double> listener) {
        trainEvaluatorListener = Optional.of(new TrainEvaluator(getNrofEvalExamples(), listener));
        getModel().addListeners(trainEvaluatorListener.get());
    }

    @Override
    public void registerValidation(Validation.Factory<? extends IEvaluation> validationFactory) {
        validations.add(validationFactory.create(evalIter.getLabels()));
    }

    @Override
    public void saveModel(String fileName) throws IOException {
        log.info("Saving model: " + name());
        ModelSerializer.writeModel(model.asModel(), new File(fileName), true);
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public double getBestEvalScore() {
        return bestEvalScore;
    }

    @Override
    public void fit() {
        if (nanTimeOutTimer < nanTimeOutTime) {
            //TODO: reload model when timer is up
            // nanTimeOutTimer++;
            log.warn("Model " + name() + " broken. Skipping...");
            return;
        }
        trainingIter.resetCursor();
        final List<INDArray> labels = new ArrayList<>();
        while (trainingIter.hasNext()) {
            labels.add(trainingIter.next().getLabels());
        }
        trainEvaluatorListener.ifPresent(te -> te.setLabels(labels));
        trainingIter.resetCursor();
        // TODO: Move into listener??
        log.info("Training model with curr best " + getBestEvalScore() + ", name: " + name());
        final long starttime = System.nanoTime();
        model.fit(trainingIter);
        final long endtime = System.nanoTime();
        final double time = (endtime - starttime) / 1000000d;
        log.info("Training took " + time + " ms for " + getNrofTrainingExamplesPerBatch() + " examples, " + time / (double) getNrofTrainingExamplesPerBatch() + " ms per example");

        trainEvaluatorListener.ifPresent(TrainEvaluator::pollListener);
    }

    @Override
    public void eval() {
        if (nanTimeOutTimer < nanTimeOutTime) {
            return;
        }

        evalIter.resetCursor();

        IEvaluation[] evalArr = validations.stream()
                .map(Validation::get)
                .filter(Optional::isPresent)
                .map(Optional::get)
                .toArray(IEvaluation[]::new);

        if (evalArr.length > 0) {
            // TODO: Move into validation??
            log.info("Begin eval of " + name());
            final long starttime = System.nanoTime();
            model.eval(evalIter, evalArr);
            final long endtime = System.nanoTime();
            final double evalTime = (endtime - starttime) / 1000000d;
            log.info("Evaluation took " + evalTime + " ms for " + getNrofEvalExamples() + " examples, " + evalTime / (double) getNrofEvalExamples() + " ms per example");
            validations.forEach(Validation::notifyComplete);
        }
    }

    @Override
    public void resetTraining() {
        trainingIter.reset();
    }
}
