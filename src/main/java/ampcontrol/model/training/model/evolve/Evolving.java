package ampcontrol.model.training.model.evolve;

/**
 * Interface for items which can be evolved
 *
 * @author Christian Skärby
 */
public interface Evolving<T extends Evolving<T>> {

    /**
     * Evolve the item
     * @return the evolved item
     */
    T evolve();
}
