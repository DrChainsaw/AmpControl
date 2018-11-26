package ampcontrol.model.training.model.evolve.crossover.state;

import ampcontrol.model.training.model.evolve.crossover.Crossover;

public class NoStateWapper<T, V> extends CrossoverState<T, V> {

    private final Crossover<T> crossover;
    private final V state;

    public NoStateWapper(Crossover<T> crossover) {
        this(crossover, null);
    }

    public NoStateWapper(Crossover<T> crossover, V state) {
        this.crossover = crossover;
        this.state = state;
    }

    @Override
    public void merge(CrossoverState<T, V> other, T thisInput, T otherInput, T result) {
        // Ignore
    }

    @Override
    public void save(String baseName) {
        // Ignore
    }

    @Override
    public CrossoverState<T, V> clone() {
        return this;
    }

    @Override
    protected V getState() {
        return state;
    }

    @Override
    public T cross(T first, T second) {
        return crossover.cross(first, second);
    }
}
