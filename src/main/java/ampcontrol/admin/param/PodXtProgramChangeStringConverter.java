package ampcontrol.admin.param;

import ampcontrol.amp.midi.program.PodXtProgramChange;
import com.beust.jcommander.IStringConverter;
import com.beust.jcommander.ParameterException;

/**
 * Returns the {@link PodXtProgramChange} with the given name.
 *
 * @author Christian Skärby
 */
public class PodXtProgramChangeStringConverter implements IStringConverter<PodXtProgramChange> {
    @Override
    public PodXtProgramChange convert(String s) {

        try {
            return PodXtProgramChange.valueOf(s);
        }catch (IllegalArgumentException e) {
            throw new ParameterException("Invalid program: " + s + "\n" + e);
        }
    }
}
