package ampcontrol.model.training.model;

import ampcontrol.model.training.model.validation.Validation;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.evaluation.IEvaluation;

import java.io.IOException;

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
     * Register a {@link Validation} to perform
     * @param validationFactory a {@link Validation.Factory} to create the validation.
     */
    void registerValidation(Validation.Factory<? extends IEvaluation> validationFactory);

    /**
     * Adds a {@link TrainingListener} to the model
     * @param listener a {@link TrainingListener}
     */
    void addListener(TrainingListener listener);

    /**
     * Serialize the model to a file with the given name.
     * @param fileName the filename
     */
    void saveModel(String fileName) throws IOException;
}
