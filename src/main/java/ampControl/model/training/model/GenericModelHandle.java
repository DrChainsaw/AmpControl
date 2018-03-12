package ampControl.model.training.model;

import ampControl.model.training.data.iterators.CachingDataSetIterator;
import ampControl.model.training.listen.NanScoreWatcher;
import ampControl.model.training.listen.TrainEvaluator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.eval.ROCMultiClass;
import org.deeplearning4j.nn.api.Model;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
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
public class GenericModelHandle implements ModelHandle {

    private static final Logger log = LoggerFactory.getLogger(GenericModelHandle.class);

    private static int nanTimeOutTime = 200;

    private final CachingDataSetIterator trainingIter;
    private final CachingDataSetIterator evalIter;
    private final ModelAdapter model;
    private final String name;
    private Optional<TrainEvaluator> trainEvaluatorListener = Optional.empty();

    private double bestEvalScore;
    private int nanTimeOutTimer = nanTimeOutTime;

    /**
     * Constructor
     * @param trainingIter Provides training samples.
     * @param evalIter Provides evaluation samples.
     * @param model Model to fit/evaluate
     * @param name Name of the model
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
        List<INDArray> labels = new ArrayList<>();
        while (trainingIter.hasNext()) {
            labels.add(trainingIter.next().getLabels());
        }
        trainEvaluatorListener.ifPresent(te -> te.setLabels(labels));
        trainingIter.resetCursor();
        model.fit(trainingIter);
        trainEvaluatorListener.ifPresent(te -> te.pollListener());
    }

    @Override
    public <T extends IEvaluation> T[] eval(T... evals) {
        if (nanTimeOutTimer < nanTimeOutTime) {
            return evals;
        }

        evalIter.resetCursor();
        Evaluation eval = null;
        for (IEvaluation evalInput : evals) {
            if (evalInput instanceof Evaluation) {
                eval = (Evaluation) evalInput;
                eval.setLabelsList(evalIter.getLabels());
            } else if (evalInput instanceof ROCMultiClass) {
                ((ROCMultiClass) evalInput).setLabels(evalIter.getLabels());
            }
        }
        T[] evalsExtra = evals;
        if (eval == null) {
            eval = createEvalTemplate();
            evalsExtra = Arrays.copyOf(evals, evals.length + 1);
            evalsExtra[evalsExtra.length - 1] = (T) eval;
        }

        model.eval(evalIter, evalsExtra);
        if (eval.accuracy() > bestEvalScore) {
            bestEvalScore = eval.accuracy();
        }

        return evals;
    }

    @Override
    public Evaluation createEvalTemplate() {
        return new Evaluation(evalIter.getLabels(), 1);
    }

    @Override
    public void resetTraining() {
        trainingIter.reset();
    }
}
