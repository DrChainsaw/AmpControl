package ampcontrol.model.training.model.evolve.selection;

import ampcontrol.model.training.model.evolve.Evolving;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * Selects a fixed predetermined number of candidates from each of a set of given {@link Selection}s
 * @param <T>
 * @author Christian Skärby
 */
public final class CompoundFixedSelection<T> implements Selection<T> {

    private final List<Selection<T>> selections;

    public static <T extends Evolving<T>> Builder<T> builder() {
        return new Builder<>();
    }

    private CompoundFixedSelection(List<Selection<T>> selections) {
        this.selections = selections;
    }

    @Override
    public Stream<T> selectCandiates(List<Map.Entry<Double, T>> fitnessCandidates) {
        return selections.stream()
                .flatMap(sel -> sel.selectCandiates(fitnessCandidates));
    }

    public static final class Builder<T> {
        private final List<Selection<T>> selections = new ArrayList<>();

        private Builder() {

        }

        public Builder<T> andThen(int nrofCandidatesToSelect, Selection<T> selection) {
            selections.add(list -> selection.selectCandiates(list).limit(nrofCandidatesToSelect));
            return this;
        }

        public CompoundFixedSelection<T> build() {
            return new CompoundFixedSelection<>(new ArrayList<>(selections));
        }

    }
}
