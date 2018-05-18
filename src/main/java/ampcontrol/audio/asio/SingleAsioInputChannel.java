package ampcontrol.audio.asio;

import com.synthbot.jasiohost.AsioChannel;

import java.util.Set;

/**
 * {@link AsioInputChannel} for an actual {@link AsioChannel}.
 *
 */
public class SingleAsioInputChannel implements AsioInputChannel {

    private final AsioChannel channel;

    public SingleAsioInputChannel(AsioChannel channel) {
        this.channel = channel;
    }

    @Override
    public boolean updateBuffer(float[] buffer, Set<AsioChannel> channels) {
        if(channels.contains(channel)) {
            channel.read(buffer);
            return true;
        }
        return false;
    }
}
