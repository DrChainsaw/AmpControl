package ampcontrol.model.training.model.evolve.selection;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * Selects candidates on order of fitness score
 * @param <T>
 *
 * @author Christian Skärby
 */
public final class EliteSelection<T> implements Selection<T> {

    private static final Logger log = LoggerFactory.getLogger(EliteSelection.class);

    @Override
    public Stream<T> selectCandiates(List<Map.Entry<Double, T>> fitnessCandidates) {
        return fitnessCandidates.stream().sorted(Map.Entry.comparingByKey())
                .peek(candEntry -> log.info("Selected cand " + fitnessCandidates.indexOf(candEntry) + " with fitness " + candEntry.getKey()))
                .map(Map.Entry::getValue);
    }
}
