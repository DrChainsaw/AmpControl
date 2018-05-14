package ampControl.model.training.model;

import ampControl.model.training.model.validation.Validation;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.nn.api.Model;

import java.io.IOException;
import java.util.function.BiConsumer;

/**
 * Interface for handling model fitting and evaluation without having to supply the dataset.
 *
 * @author Christian Skärby
 */
public interface ModelHandle {

    /**
     * Performs a round of fitting. Exactly what a "round" is depends on implementation.
     */
    void fit();

    /**
     * Evaluates the model
     */
    void eval();

    /**
     * Prepares for another "round" of fitting. Separated from fit as models might (and typically do) share training
     * data.
     */
    void resetTraining();

    /**
     * Returns the name of the model
     * @return the name of the model
     */
    String name();

    /**
     * Returns the best evaluation score so far. Higher is better.
     * @return the best evaluation score so far
     */
    double getBestEvalScore();

    /**
     * Returns the {@link Model}
     * @return the {@link Model}
     */
    Model getModel();

    /**
     * Returns the number of batches in each training round.
     * @return the number of batches in each training round.
     */
    int getNrofBatchesForTraining();

    /**
     * Returns the number of examples per batch.
     * @return the number of examples per batch.
     */
    int getNrofTrainingExamplesPerBatch();

    /**
     * Returns the number of examples for evaluation
     * @return the number of examples for evaluation
     */
    int getNrofEvalExamples();

    /**
     * Create a listener for training evaluation. Will typically slow down training by a few percent.
     * @param accuracyCallback Will be notified of accuracy per iteration.
     */
    void createTrainingEvalListener(BiConsumer<Integer, Double> accuracyCallback);

    /**
     * Register a {@link Validation} to perform
     * @param validationFactory a {@link Validation.Factory} to create the validation.
     */
    void registerValidation(Validation.Factory<? extends IEvaluation> validationFactory);

    /**
     * Serialize the model to a file with the given name.
     * @param fileName the filename
     */
    void saveModel(String fileName) throws IOException;
}
