package ampcontrol.model.training.model.evolve.selection;

import ampcontrol.model.training.model.evolve.Evolving;

import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * Evolves selected candidates
 * @param <T>
 *
 * @author Christian Skärby
 */
public final class EvolveSelection<T extends Evolving<T>> implements Selection<T> {

    private final Selection<T> sourceSelection;

    public EvolveSelection(Selection<T> sourceSelection) {
        this.sourceSelection = sourceSelection;
    }

    @Override
    public Stream<T> selectCandiates(List<Map.Entry<Double, T>> fitnessCandidates) {
        return sourceSelection.selectCandiates(fitnessCandidates).map(Evolving::evolve);
    }
}
