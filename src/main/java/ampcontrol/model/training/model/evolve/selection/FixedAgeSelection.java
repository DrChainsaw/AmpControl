package ampcontrol.model.training.model.evolve.selection;

import ampcontrol.model.training.model.CompGraphAdapter;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Removes candidates which have exceeded a given max age. This regularizes evolutionary algorithms. Concept from
 * https://arxiv.org/abs/1802.01548.
 * @param <T>
 *
 * @author Christian Skärby
 */
public class FixedAgeSelection<T> implements Selection<T> {

    private final Function<T, Object> toHashable;
    private final Selection<T> sourceSelection;
    private final int maxAge;
    private final Map<Object, Integer> ageMap = new HashMap<>();

    /**
     * Constructor
     * @param toHashable Function to turn a candidate into a hashable object. Allows for different individuals to still
     *                   be considered the same, e.g. if the an individual is mutated but mutation did not change anything
     * @param sourceSelection Source selection. Will not see any candidates older than maxAge
     * @param maxAge Candidates seen more than this number of times will be removed
     */
    public FixedAgeSelection(Function<T, Object> toHashable, Selection<T> sourceSelection, int maxAge) {
        this.toHashable = toHashable;
        this.sourceSelection = sourceSelection;
        this.maxAge = maxAge;
    }

    @Override
    public Stream<T> selectCandiates(List<Map.Entry<Double, T>> fitnessCandidates) {
        final Map<T, Object> hashables = fitnessCandidates.stream()
                .map(Map.Entry::getValue)
                .collect(Collectors.toMap(
                        t -> t,
                        toHashable
                ));

        ageMap.keySet().retainAll(hashables.values());

        return sourceSelection.selectCandiates(
                fitnessCandidates.stream()
                        .peek(entry -> updateAge(hashables.get(entry.getValue())))
                        .filter(entry -> ageMap.get(hashables.get(entry.getValue())) < maxAge)
                        .collect(Collectors.toList()));
    }

    private void updateAge(Object candidate) {
        ageMap.computeIfPresent(candidate, (cand, age) -> age + 1);
        ageMap.putIfAbsent(candidate, 0);
    }

    /**
     * Creates a {@link FixedAgeSelection} where two individuals are considered the same for ageing purposes if they have the
     * exact same ComputationGraphConfiguration.
     * @param maxAge Maximum age
     * @param sourceSelection source selection
     * @return a new FixedAgeSelection
     */
    public static <T extends CompGraphAdapter> FixedAgeSelection<T> byConfig(int maxAge, Selection<T> sourceSelection) {
        return new FixedAgeSelection<>(
                adapter -> Stream.of(adapter.asModel().getConfiguration().toYaml().split("\n"))
                        .filter(line -> !line.contains("seed"))
                        .collect(Collectors.joining("\n")),
                sourceSelection,
                maxAge
        );
    }
}
