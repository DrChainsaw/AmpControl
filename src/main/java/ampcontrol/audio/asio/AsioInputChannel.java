package ampcontrol.audio.asio;

import com.synthbot.jasiohost.AsioChannel;

import java.util.Set;

/**
 * Interface for asio channel input. Main purpose is to allow workarounds as {@link AsioChannel} is not mockable.
 *
 * @author Christian Skärby
 */
public interface AsioInputChannel {

    /**
     * Updates the given buffer if applicable. Returns true if the buffer was updated.
     *
     * @param buffer
     * @param channels
     * @return true if the given buffer was updated
     */
    boolean updateBuffer(float[] buffer, Set<AsioChannel> channels);


}
