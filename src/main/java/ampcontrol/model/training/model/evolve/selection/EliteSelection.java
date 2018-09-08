package ampcontrol.model.training.model.evolve.selection;

import ampcontrol.model.training.model.evolve.Evolving;

import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * Selects candidates on order of fitness score
 * @param <T>
 *
 * @author Christian Skärby
 */
public class EliteSelection<T extends Evolving<T>> implements Selection<T> {

    @Override
    public Stream<T> selectCandiates(List<Map.Entry<Double, T>> fitnessCandidates) {
        return fitnessCandidates.stream().sorted(Map.Entry.comparingByKey()).map(Map.Entry::getValue);
    }
}
