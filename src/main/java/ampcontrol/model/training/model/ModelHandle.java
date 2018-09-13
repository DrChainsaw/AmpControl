package ampcontrol.model.training.model;

import ampcontrol.model.training.model.validation.Validation;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.nn.api.Model;

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
     * Returns the {@link Model}
     * @return the {@link Model}
     */
    Model getModel();

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

    /**
     * Compresses the string into a shorter version if it is "too long"
     * @param str String to compress
     * @return Compressed version of the string
     */
    static String compressIfNeeded(String str) {
        if(str.length() > 100) {
            return String.valueOf(str.hashCode());
        }
        return str;
    }
}
