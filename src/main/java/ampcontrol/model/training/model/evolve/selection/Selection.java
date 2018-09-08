package ampcontrol.model.training.model.evolve.selection;

import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * Interface for selecting candidates from a list of fitness and candidate pairs
 * @param <T>
 *
 * @author Christian Skärby
 */
public interface Selection<T> {
    /**
     * Selects candidates from the population
     * @param fitnessCandidates List of candidates and their fitness values
     * @return A stream of selected candidates.
     */
    Stream<T> selectCandiates(List<Map.Entry<Double, T>> fitnessCandidates);
}
