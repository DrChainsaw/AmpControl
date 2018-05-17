package ampcontrol.amp.labelmapping;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;

/**
 * Generic {@link LabelMapping} which masks out given labels from a given {@link LabelMapping}.
 *
 * @param <T>
 *
 * @author Christian Skärby
 */
public class MaskingLabelMapping<T> implements LabelMapping<T> {

    private final LabelMapping<T> sourceMapper;
    private final BitSet labelMask = new BitSet();


    public MaskingLabelMapping(List<Integer> labelMask, LabelMapping<T> sourceMapper) {
        this.sourceMapper = sourceMapper;
        for(int labelToHold: labelMask) {
           this.labelMask.set(labelToHold);
        }

    }

    @Override
    public List<T> apply(Integer labelInd) {
        if(labelMask.get(labelInd)) {
            return new ArrayList<>();
        }
        return sourceMapper.apply(labelInd);
    }
}
