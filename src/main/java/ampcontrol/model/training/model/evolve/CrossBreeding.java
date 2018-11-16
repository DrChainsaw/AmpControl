package ampcontrol.model.training.model.evolve;

/**
 * Interface for items which can crossbreed
 * @param <T>
 * @author Christian Skärby
 */
public interface CrossBreeding<T> {

    /**
     * Crossbreed the item with the given mate
     * @param mate
     * @return the offspring
     */
    T cross(T mate);

}
