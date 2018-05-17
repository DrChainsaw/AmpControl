package ampcontrol.model.inference;

import ampcontrol.audio.ClassifierInputProvider;

import java.io.IOException;

/**
 * Factory for creating a {@link Classifier} from a string, typically a filename.
 */
public interface ClassifierFactory {

    /**
     * Create a {@link Classifier}
     *
     * @param path
     * @param inputProvider
     * @return  a {@link Classifier}
     */
    Classifier create(String path, final ClassifierInputProvider inputProvider) throws IOException;
}
