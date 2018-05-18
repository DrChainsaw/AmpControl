package ampcontrol.amp.labelmapping;

import java.util.List;
import java.util.function.Function;

/**
 * Maps a label index to a list of something
 *
 * @param <T> Type to map the label to
 *
 * @author Christian Skärby
 */
public interface LabelMapping<T> extends Function<Integer, List<T>> {
    // Just an alias
}
