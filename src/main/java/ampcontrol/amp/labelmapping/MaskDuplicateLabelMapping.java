package ampcontrol.amp.labelmapping;

import java.util.ArrayList;
import java.util.List;

/**
 * Generic {@link LabelMapping} to mask duplicates from a provided {@link LabelMapping}.
 * Typical use case is to prevent that multiple identical program changes are sent to a MIDI device.
 *
 * @param <T>
 *
 * @author Christian Skärby
 */
public class MaskDuplicateLabelMapping<T> implements LabelMapping<T> {

    private final LabelMapping<T> sourceLabelMapping;
    private int lastLabel = -1;

    public MaskDuplicateLabelMapping(LabelMapping<T> sourceLabelMapping) {
        this.sourceLabelMapping = sourceLabelMapping;
    }

    @Override
    public List<T> apply(Integer labelInd) {
        if(lastLabel == labelInd) {
            return new ArrayList<>();
        }

        List<T> ret = sourceLabelMapping.apply(labelInd);
        if(!ret.isEmpty()) {
            lastLabel = labelInd;
        }
        return ret;
    }
}
