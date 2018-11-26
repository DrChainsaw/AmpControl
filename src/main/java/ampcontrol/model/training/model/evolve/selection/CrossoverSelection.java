package ampcontrol.model.training.model.evolve.selection;

import ampcontrol.model.training.model.evolve.CrossBreeding;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Performs crossover between selected candidates.
 * @param <T>
 * @author Christian Skärby
 */
public class CrossoverSelection<T extends CrossBreeding<T>> implements Selection<T> {

    private final Predicate<T> crossoverProbability;
    private final BiFunction<T, List<T>, T> selectMate;
    private final Selection<T> sourceSelection;

    public CrossoverSelection(Predicate<T> crossoverProbability, BiFunction<T, List<T>, T> selectMate, Selection<T> sourceSelection) {
        this.crossoverProbability = crossoverProbability;
        this.selectMate = selectMate;
        this.sourceSelection = sourceSelection;
    }

    @Override
    public Stream<T> selectCandiates(List<Map.Entry<Double, T>> fitnessCandidates) {
        final List<T> selectedCands = sourceSelection.selectCandiates(fitnessCandidates).collect(Collectors.toList());
        final List<T> processedCands = new ArrayList<>();
        for(int i = 0; i < selectedCands.size(); i++) {
            T cand = selectedCands.get(i);
            if(crossoverProbability.test(cand)) {
                T newCand = cand.cross(selectMate.apply(cand, selectedCands));
                processedCands.add(newCand);
            } else {
                processedCands.add(cand);
            }
        }
        return processedCands.stream();
    }
}
