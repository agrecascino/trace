#ifndef PLAYER_H
#define PLAYER_H
#include <iostream>
#include <fstream>
#include <map>
#include <portaudio.h>
#include <math.h>
#include <cstdlib>
#include <string.h>
#include <stdint.h>
#include <assert.h>


struct Sample {
    Sample() {
        finetune = 0;
    }

    std::string name;
    uint64_t length = 0;
    uint8_t finetune : 4;
    uint8_t volume = 0;
    uint64_t loopstart = 0;
    uint64_t looplength = 0;
    int8_t *data;
};

struct Note {
    Note() {
        period = 0;
        sample = 0;
        effect = 0;
        argument = 0;
    }

    uint16_t period : 12;
    uint8_t sample;
    uint8_t effect : 4;
    uint8_t argument;
};

struct Row {
    uint64_t nchannels;
    Note *notes;
};

struct Pattern {
    uint64_t nrows;
    Row *rows;
};

struct Module {
    std::string name;
    Sample *samples;
    uint64_t nsamples;
    uint64_t npatterns;
    uint64_t norders;
    uint8_t *orders;
    Pattern *patterns;
};

enum TrackerEffectQuirks {
    EFXISPANNING = 0x00000001
};

struct TrackerQuirks {
    TrackerQuirks() {}
    TrackerQuirks(uint64_t nchannels, uint32_t effectquirks) :
        nchannels(nchannels) , effectquirks(effectquirks) {}
    uint64_t nchannels = 4;
    uint32_t effectquirks = 0;
};

enum ModuleLoadState {
    LOAD_FAILED_HEADER,
    LOAD_FAILED_PATTERN,
    LOAD_FAILED_OTHER,
    LOAD_FAILED_SAMPLE,
    LOAD_OK
};

enum Verbosity {
    NONE,
    MESSAGE,
    DEBUG
};

struct ChannelState {
    ChannelState() {
        latchedperiod = 0;
        lasteffect = 0;
        liveperiod = 0;
        liveeffect = 0;
    }

    double samplepoint = 0.0;
    unsigned short latchedperiod : 12;
    unsigned char  latchedsample = 0;
    unsigned char  latchedvolume = 0;
    unsigned char  lasteffectparam = 0;
    unsigned char  lasteffect : 4;
    unsigned char  livesample = 0;
    unsigned short liveperiod : 12;
    unsigned char  livevolume = 0;
    unsigned char  liveeffect : 4;
    unsigned char  liveeffectparam = 0;
    unsigned int   offset = 0;
    bool ploop = 0;
    unsigned int   loopstartrow = 0;
    unsigned int   loopcnt = 0;
    unsigned int   portamemory = 0;
    unsigned int   offsetmemory = 0;
};

struct TrackerState {
    ChannelState *cstate;
    uint64_t tpr = 6;
    uint64_t bpm = 125;
    uint64_t samplerate = 44100;
    uint64_t SamplesPerTick() {
        return samplerate * (2500.0/bpm)/1000.0;
    }
};

enum ReturnAction {
    TICK,
    INC,
    JUMP
};

struct JumpLocation {
    uint64_t order;
    uint64_t row;
};

struct TickReturn {
    int16_t *audio[2];
    uint64_t nsamples;
    ReturnAction action;
    JumpLocation location;
};

enum PlayReturn {
    PLAY_OK,
    PLAY_FAILED
};

struct TickState {
    std::map<int, bool> effectseen;
};

class PeriodCorrector {
public:
    PeriodCorrector() {
        GeneratePTPeriodTable(periods);
    }

    unsigned short CorrectPeriod(unsigned short period, unsigned char finetune) {
        for(int i = 0; i < 36; i++) {
            if(periods[0][i] == period)
                return periods[finetune][i];
        }
        return 0;
    }

private:
    double pow2(double x) {
        return pow(2.0, x);
    }

    void GeneratePTPeriodTable(unsigned short (*periods)[36]) {
        const double NTSC_CLK        = 3579545.0;
        const double REF_PERIOD_PT   = 856.0;
        const double REF_PERIOD_UST  = NTSC_CLK / 523.3 / 8;
        const double UST_TO_PT_RATIO = REF_PERIOD_UST / REF_PERIOD_PT;
        const double semitone_step   = pow2(-1.0/12.0);
        const double tune_step       = pow2(-1.0/8.0 * 1.0/12.0);
        int n, t;
        // Initialize with starting period, i.e. 907
        double p1 = REF_PERIOD_PT / semitone_step;
        for(t = 0 ; t < 8 ; t++) {
            // Initialize with starting period for current tuning
            double p2 = p1;
            for(n = 0 ; n < 36 ; n++) {
                // Round and save current period, update period for next Semitone
                periods[t+8][n]   = (unsigned short)(p2 + 0.5);
                p2               *= semitone_step;
                periods[t][n]     = (unsigned short)(p2 + 0.5);
                // Save correct UST period for normal tuning
                if(t == 0) {
                    periods[0][n] = (unsigned short)(p2 * UST_TO_PT_RATIO + 0.5);
                }
            }
            // Starting period for next tuning
            p1 *= tune_step;
        }
        // Create correct values for the octave halved periods for normal tuning
        for(n = 0 ; n < 9 ; n++)   { periods[0][n] = periods[0][n+12] * 2; }
        // Copy UST periods to tuning -8
        for(n = 1 ; n < 36 ; n++)  { periods[8][n] = periods[0][n-1];      }
        // Correct those 9 #?!$?#!%!! entries that refuse
        periods[1][ 4]--;  periods[1][22]++;  periods[ 1][24]++;
        periods[2][23]++;  periods[4][ 9]++;  periods[ 7][24]++;
        periods[9][ 6]--;  periods[9][26]--;  periods[12][34]--;
    }

    unsigned short periods[16][36];
};

class ModulePlayer {
public:
    ModulePlayer(std::fstream &moduledata, Verbosity verbosity = MESSAGE) : verbosity(verbosity) {
        Pa_Initialize();
        Pa_OpenDefaultStream(&stream, 0, 2, paInt16, 44100.0, paFramesPerBufferUnspecified, NULL, NULL);
        Pa_StartStream(stream);
        if(!LoadModuleHeader(moduledata)) {
            loadstate = LOAD_FAILED_HEADER;
            return;
        }
        if(!LoadSampleHeaders(moduledata)) {
            loadstate = LOAD_FAILED_SAMPLE;
            return;
        }
        if(!LoadPatternsAndOrders(moduledata)) {
            loadstate = LOAD_FAILED_PATTERN;
            return;
        }
        if(!LoadSampleData(moduledata)){
            loadstate = LOAD_FAILED_SAMPLE;
            return;
        }
        state.cstate = new ChannelState[mod.patterns[0].rows[0].nchannels];
        loadstate = LOAD_OK;
    }

    PlayReturn playModule() {
        if(verbosity > NONE) {
            switch(loadstate) {
            case LOAD_OK:
                std::cout << "Module load was successful, starting playback!"
                          << std::endl;
                break;
            case LOAD_FAILED_HEADER:
                std::cout << "Module Load failed at header, is this a MOD file?"
                          << std::endl;
                return PLAY_FAILED;
            case LOAD_FAILED_PATTERN:
                std::cout << "Module load failed at pattern loading, module may be corrupted."
                          << std::endl;
                return PLAY_FAILED;
            case LOAD_FAILED_SAMPLE:
                std::cout << "Module load failed at sample loading, module may be corrupted."
                          << std::endl;
                return PLAY_FAILED;
            case LOAD_FAILED_OTHER:
                std::cout << "Module load failed in an unknown way. Oh no."
                          << std::endl;
                return PLAY_FAILED;
            }
        }
        uint64_t row = 0;
        uint64_t tick = 0;
        for(uint64_t i = 0; i < mod.norders; i++) {
            while(true) {
                lastrow = row;
                lastorder = i;
                TickReturn ret = PlayOneTick(i, row, tick);
                int16_t *audio = new int16_t[ret.nsamples*2];
                for(uint64_t s = 0; s < ret.nsamples; s++) {
                    audio[s*2] = ret.audio[0][s] + ret.audio[1][s]*0.5;
                    audio[(s*2) + 1] = ret.audio[1][s]  + ret.audio[0][s]*0.5;
                }
                delete[] ret.audio[0];
                delete[] ret.audio[1];
                Pa_WriteStream(stream, audio, ret.nsamples);
                delete[] audio;
                switch(ret.action) {
                case TICK:
                    tick++;
                    break;
                case INC:
                    tick = 0;
                    if(row == 63) {
                        row = 0;
                        goto nextorder;
                    }
                    row++;
                    break;
                case JUMP:
                    tick = 0;
                    i = ret.location.order - 1;
                    row = ret.location.row;
                    goto nextorder;
                }
            }
nextorder:
            continue;
        }
        return PLAY_OK;
    }

    TickReturn PlayOneTick(uint64_t order, uint64_t row, uint8_t tick) {
        TickReturn ret;
        ret.action = TICK;
        if(tick == (state.tpr - 1)) {
            ret.action = INC;
        }
        ret.audio[0] = new int16_t[state.SamplesPerTick()];
        memset(ret.audio[0], 0, state.SamplesPerTick()*2);
        ret.audio[1] = new int16_t[state.SamplesPerTick()];
        memset(ret.audio[1], 0, state.SamplesPerTick()*2);
        ret.nsamples = state.SamplesPerTick();
        TickState ts;
        for(uint64_t i = 0; i < mod.patterns[mod.orders[order]].rows[0].nchannels; i++) {
            Note n = mod.patterns[mod.orders[order]].rows[row].notes[i];
            if(tick == (state.tpr - 1)) {
                //Effects performed on last tick
                switch(n.effect) {
                case 0xD:
                    if(ts.effectseen[0xE6])
                        break;
                    ts.effectseen[0xD] = true;
                    ret.action = JUMP;
                    if(!ts.effectseen[0xB])
                        ret.location.order = order + 1;
                    ret.location.row = (10 * (n.argument & 0xF0 >> 4)) + (n.argument & 0x0F);
                    if(ret.location.row > 63)
                        ret.location.row = 63;
                    break;
                case 0xB:
                    if(ts.effectseen[0xE6])
                        break;
                    ts.effectseen[0xB] = true;
                    ret.action = JUMP;
                    ret.location.order = n.argument;
                    if(ret.location.order >  mod.norders)
                        ret.location.order = 0;
                    if(!ts.effectseen[0xD])
                        ret.location.row = 0;
                    break;
                case 0xE:
                    switch(n.argument >> 4) {
                    case 0x6:
                        if(!(n.argument & 0x0F))
                            state.cstate[i].loopstartrow = row;
                        else {
                            if(!state.cstate[i].loopcnt)
                                state.cstate[i].loopcnt = (n.argument & 0x0F) + 1;
                            if((state.cstate[i].loopcnt - 1)) {
                                state.cstate[i].loopcnt--;
                                ts.effectseen[0xE6] = true;
                                ret.action = JUMP;
                                ret.location.order = order - 1;
                                ret.location.row = state.cstate[i].loopstartrow;
                            }
                        }
                        break;

                    }
                    break;
                }
            }
            if(!tick) {
                std::cout << n.period;
                std::cout << " ";
                std::cout << (int)n.sample;
                std::cout << " ";
                if(n.sample != 0) {
                    state.cstate[i].latchedsample = n.sample;
                    state.cstate[i].latchedvolume = mod.samples[n.sample - 1].volume;
                    state.cstate[i].livevolume = mod.samples[n.sample - 1].volume;
                }
                bool unhit = false;

                if(n.period != 0) {
                    if(!state.cstate[i].livesample) {
                        if(state.cstate[i].latchedsample)
                            state.cstate[i].livesample = state.cstate[i].latchedsample;
                        else
                            goto nosample;
                    } else if(n.sample)
                        state.cstate[i].livesample = n.sample;
                    state.cstate[i].latchedperiod = corrector.CorrectPeriod(n.period, mod.samples[state.cstate[i].livesample - 1].finetune);
                    if(((n.effect != 0x3) && (n.effect != 0x5)) || (((n.effect == 0x3) || (n.effect == 0x5)) && !state.cstate[i].liveperiod)) {
                        state.cstate[i].samplepoint = 0;
                        state.cstate[i].liveperiod = state.cstate[i].latchedperiod;
                    }
                }
                if(unhit) {
                    std::cout << (int)state.cstate[i].livesample << std::endl;
                    std::cout << state.cstate[i].samplepoint << std::endl;
                    std::cout << state.cstate[i].liveperiod << std::endl;
                }
nosample:
                //Effects that are performed once a row, on the zero tick.
                switch(n.effect) {
                case 0xC:
                    state.cstate[i].livevolume = n.argument;
                    state.cstate[i].latchedvolume = n.argument;
                    break;
                case 0xF:
                    if(!n.argument)
                        break;
                    if(n.argument < 32)
                        state.tpr = n.argument;
                    else
                        state.bpm = n.argument;
                    break;
                case 0xE:
                    switch((n.argument & 0xF0 >> 4)) {
                    case 1:
                        state.cstate[i].liveperiod -= n.argument & 0x0F;
                        if(state.cstate[i].liveperiod < 113)
                            state.cstate[i].liveperiod = 113;
                        break;
                    case 2:
                        state.cstate[i].liveperiod += n.argument & 0x0F;
                        if(state.cstate[i].liveperiod > 856)
                            state.cstate[i].liveperiod = 856;
                        break;
                    }
                    break;
                case 0x9:
                    if(!n.argument) {
                        state.cstate[i].samplepoint = state.cstate[i].offsetmemory;
                        break;
                    }
                    state.cstate[i].offsetmemory = n.argument * 256;
                    state.cstate[i].samplepoint = n.argument * 256;
                    break;
                }
                } else {
                switch(n.effect) {
                case 1:
                    state.cstate[i].liveperiod -= n.argument;
                    if(state.cstate[i].liveperiod < 113)
                        state.cstate[i].liveperiod = 113;
                    break;
                case 2:
                    state.cstate[i].liveperiod += n.argument;
                    if(state.cstate[i].liveperiod > 856)
                        state.cstate[i].liveperiod = 856;
                    break;
                case 3: {
                    if(!n.argument)
                        n.argument = state.cstate[i].portamemory;
                    else
                        state.cstate[i].portamemory = n.argument;
                    if(state.cstate[i].latchedperiod == state.cstate[i].liveperiod)
                        break;
                    float direction = copysign(1, state.cstate[i].latchedperiod - state.cstate[i].liveperiod);
                    state.cstate[i].liveperiod = state.cstate[i].liveperiod + (direction * n.argument);
                    float direction2 = copysign(1, state.cstate[i].latchedperiod - state.cstate[i].liveperiod);
                    if((int)direction != (int)direction2)
                        state.cstate[i].liveperiod = state.cstate[i].latchedperiod;
                    break; }
                case 5: {
                    if(state.cstate[i].latchedperiod == state.cstate[i].liveperiod)
                        break;
                    float direction = copysign(1, state.cstate[i].latchedperiod - state.cstate[i].liveperiod);
                    state.cstate[i].liveperiod = state.cstate[i].liveperiod + (direction * state.cstate[i].portamemory);
                    float direction2 = copysign(1, state.cstate[i].latchedperiod - state.cstate[i].liveperiod);
                    if((int)direction != (int)direction2)
                        state.cstate[i].liveperiod = state.cstate[i].latchedperiod;
                    break; }
                case 0xA:
                    if(!n.argument)
                        break;
                    unsigned char x = (n.argument & 0xF0) >> 4;
                    unsigned char y = (n.argument & 0x0F);
                    if(x && y)
                        break;
                    int cv = state.cstate[i].livevolume;
                    if(x)
                        cv += x;
                    if(y)
                        cv -= y;
                    cv = fmaxf(fminf(cv, 64), 0);
                    state.cstate[i].livevolume = cv;
                    break;


                }
            }

            for(uint64_t sample = 0; sample < state.SamplesPerTick(); sample++) {
                if(state.cstate[i].livesample && (state.cstate[i].livesample <= mod.nsamples)) {
                    if((state.cstate[i].samplepoint > (mod.samples[state.cstate[i].livesample - 1].loopstart + mod.samples[state.cstate[i].livesample - 1].looplength))
                            && (mod.samples[state.cstate[i].livesample - 1].looplength > 3)) {
                        state.cstate[i].samplepoint = mod.samples[state.cstate[i].livesample - 1].loopstart;
                    }
                    if((uint64_t)state.cstate[i].samplepoint >= (mod.samples[state.cstate[i].livesample - 1].length)) {
                        state.cstate[i].liveperiod = 0;
                        state.cstate[i].samplepoint = 0;
                        state.cstate[i].livesample = 0;
                    }
                    if(state.cstate[i].liveperiod) {
                        /*if(i == 2) {44100.0
                            Sample s = mod.samples[state.cstate[i].livesample - 1];
                            double a = s.data[(uint64_t)state.cstate[i].samplepoint];
                            ret.audio[0][sample] += a;
                        }*/
                        Sample s = mod.samples[state.cstate[i].livesample - 1];
                        double a = s.data[(uint64_t)state.cstate[i].samplepoint];
                        assert((uint64_t)state.cstate[i].samplepoint < s.length);
                        ret.audio[i % 2][sample] += ((a*256)*(state.cstate[i].livevolume/64.0))/mod.patterns[mod.orders[order]].rows[0].nchannels*2;
                        state.cstate[i].samplepoint += (7093789.2/(state.cstate[i].liveperiod*2))/44100.0;
                        //state.cstate[i].samplepoint += 44100.0/(7093789.2 / state.cstate[i].liveperiod*2);
                        //state.cstate[i].samplepoint += 0.03125;
                    }
                }
            }
        }
        if(!tick)
            std::cout << std::endl;
        return ret;
    }

    std::atomic<uint64_t> lastrow;
    std::atomic<uint64_t> lastorder;

private:
    int LoadSampleData(std::fstream &moduledata) {
        for(uint64_t i = 0; i < mod.nsamples; i++) {
            if(!mod.samples[i].length)
                continue;
            mod.samples[i].data = new int8_t[mod.samples[i].length];
            moduledata.read((char*)mod.samples[i].data, mod.samples[i].length);
            if(!moduledata.good()) {
                return 0;
            }
        }
        return 1;
    }

    int LoadModuleHeader(std::fstream &moduledata) {
        bool soundtracker = false;
        moduledata.seekg(1080);
        std::string signature;
        for(int i = 0; i < 4; i++) {
            signature += moduledata.get();
        }
        if(!moduledata.good()) {
            return 0;
        }
        for(char c : signature) {
            if((c < 32) && (c > 126)) {
                soundtracker = true;
                break;
            }
        }
        moduledata.seekg(0);
        for(int i = 0; i < 20; i++) {
            mod.name = moduledata.get();
        }
        if(!moduledata.good()) {
            return 0;
        }
        mod.nsamples = soundtracker ? 15 : 31;
        return 1;
    }

    int LoadSampleHeaders(std::fstream &moduledata) {
        mod.samples = new Sample[mod.nsamples];
        for(uint64_t i = 0; i < mod.nsamples; i++) {
            for(int z = 0; z < 22; z++) {
                mod.samples[i].name = moduledata.get();
            }
            mod.samples[i].length |= (unsigned long)moduledata.get() << 8;
            mod.samples[i].length |= moduledata.get();
            mod.samples[i].length *= 2;
            std::cout << mod.samples[i].length << std::endl;
            mod.samples[i].finetune = moduledata.get();
            mod.samples[i].volume = moduledata.get();
            mod.samples[i].loopstart |= (unsigned long)moduledata.get() << 8;
            mod.samples[i].loopstart |= moduledata.get();
            mod.samples[i].loopstart *= 2;
            mod.samples[i].looplength |= (unsigned long)moduledata.get() << 8;
            mod.samples[i].looplength |= moduledata.get();
            mod.samples[i].looplength *= 2;
            if(!moduledata.good()) {
                return 0;
            }
        }
        return 1;
    }

    void GenerateFastAndTakeTrackerChannelDefinitions(std::map<std::string, TrackerQuirks> &quirks) {
        for(int i = 10; i < 33; i++) {
            if(i % 2 == 0) {
                quirks[std::to_string(i) + "CN"] = TrackerQuirks(i, 0);
                quirks[std::to_string(i) + "CH"] = TrackerQuirks(i, 0);
            }
        }
    }

    int LoadPatternsAndOrders(std::fstream &moduledata) {
        mod.norders = moduledata.get();
        mod.orders = new uint8_t[mod.norders];
        moduledata.get();
        //Restart point, unsure of what to do with it.
        mod.npatterns = 0;
        for(uint64_t i = 0; i < 128; i++){
            if(i < mod.norders) {
                mod.orders[i] = moduledata.get();
                mod.npatterns = mod.npatterns > (uint64_t)(mod.orders[i] + 1) ? (mod.npatterns) : (mod.orders[i] + 1);

            } else {
                uint8_t item = moduledata.get();
                mod.npatterns = mod.npatterns > (uint64_t)(item + 1) ? (mod.npatterns) : (item + 1);
            }
        }
        if(!moduledata.good()) {
            return 0;
        }
        std::string sampletag = "";
        for(int i = 0; i < 4; i++) {
            sampletag += moduledata.get();
        }
        if(!moduledata.good()) {
            return 0;
        }
        uint64_t nchannels = 0;
        std::map<std::string, TrackerQuirks> tag;
        tag["6CHN"] = TrackerQuirks(6, 0);
        tag["8CHN"] = TrackerQuirks(8, 0);
        tag["OCTA"] = TrackerQuirks(8, 0);
        tag["OKTA"] = TrackerQuirks(8 ,0);
        tag["CD81"] = TrackerQuirks(8, 0);
        tag["TDZ1"] = TrackerQuirks(1, 0);
        tag["TDZ2"] = TrackerQuirks(2, 0);
        tag["TDZ3"] = TrackerQuirks(3, 0);
        tag["5CHN"] = TrackerQuirks(5, 0);
        tag["7CHN"] = TrackerQuirks(7, 0);
        tag["9CHN"] = TrackerQuirks(9, 0);
        GenerateFastAndTakeTrackerChannelDefinitions(tag);
        nchannels = tag[sampletag].nchannels;
        mod.patterns = new Pattern[mod.npatterns];
        for(uint8_t i = 0; i < mod.npatterns; i++) {
            mod.patterns[i].nrows = 64;
            mod.patterns[i].rows = new Row[64];
            for(uint64_t row = 0; row < mod.patterns[i].nrows; row++){
                mod.patterns[i].rows[row].nchannels = nchannels;
                mod.patterns[i].rows[row].notes = new Note[nchannels];
                for(uint64_t channel = 0; channel < nchannels; channel++) {
                    uint8_t note[4];
                    note[0] = (unsigned long)moduledata.get();
                    note[1] = (unsigned long)moduledata.get();
                    note[2] = (unsigned long)moduledata.get();
                    note[3] = (unsigned long)moduledata.get();
                    Note stnote;
                    stnote.period = ((note[0] & 0x0F) << 8) | (note[1]);
                    stnote.effect = note[2] & 0x0F;
                    stnote.argument = note[3];
                    stnote.sample = (note[0] & 0xF0) | (note[2] & 0xF0) >> 4;
                    mod.patterns[i].rows[row].notes[channel] = stnote;
                }
            }
        }
        if(!moduledata.good()) {
            return 0;
        }
        return 1;
    }
    PaStream *stream;
    PeriodCorrector corrector;
    Verbosity verbosity = NONE;
    TrackerState state;
    ModuleLoadState loadstate = LOAD_FAILED_OTHER;
    Module mod;
};

#endif // PLAYER_H
