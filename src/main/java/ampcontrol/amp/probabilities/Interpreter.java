package ampcontrol.amp.probabilities;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.function.Function;

/**
 * Facade like interface for mapping an {@link INDArray} of probabilities to a list of something, e.g MIDI commands.
 *
 * @param <T>
 */
public interface Interpreter<T> extends Function<INDArray, List<T>> {

}
