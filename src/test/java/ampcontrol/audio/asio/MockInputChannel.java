package ampcontrol.audio.asio;

import com.synthbot.jasiohost.AsioChannel;

import java.util.Set;

/**
 * Mock implementation of {@link AsioInputChannel} for testing
 *
 */
public class MockInputChannel implements AsioInputChannel {

    private float[] newBuffer;

    @Override
    public boolean updateBuffer(float[] buffer, Set<AsioChannel> channels) {
        if(buffer.length != newBuffer.length) {
            throw new IllegalArgumentException("Must be same length!");
        }
        System.arraycopy(newBuffer, 0, buffer, 0, buffer.length);
        return true;
    }

    void setNewBuffer(float[] newBuffer) {
        this.newBuffer = newBuffer;
    }
}
