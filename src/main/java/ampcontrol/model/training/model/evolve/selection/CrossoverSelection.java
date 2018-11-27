package ampcontrol.model.training.model.evolve.selection;

import ampcontrol.model.training.model.evolve.CrossBreeding;

import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Performs crossover between selected candidates.
 *
 * @param <T>
 * @author Christian Skärby
 */
public class CrossoverSelection<T extends CrossBreeding<T>> implements Selection<T> {

    private final BiFunction<T, List<T>, T> selectMate;
    private final Selection<T> sourceSelection;

    public CrossoverSelection(BiFunction<T, List<T>, T> selectMate, Selection<T> sourceSelection) {
        this.selectMate = selectMate;
        this.sourceSelection = sourceSelection;
    }

    @Override
    public Stream<T> selectCandiates(List<Map.Entry<Double, T>> fitnessCandidates) {
        final List<T> cands = fitnessCandidates.stream().map(Map.Entry::getValue).collect(Collectors.toList());
        return sourceSelection.selectCandiates(fitnessCandidates)
                .map(cand ->  cand.cross(selectMate.apply(cand, cands)));
    }
}
