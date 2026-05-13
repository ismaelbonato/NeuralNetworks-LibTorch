#include "FIRFilterModelJson.h"

#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace audio::firFilter {

namespace {

struct JsonValue
{
    enum class Type
    {
        Object,
        Array,
        String,
        Number
    };

    Type type = Type::Object;
    std::map<std::string, JsonValue> object;
    std::vector<JsonValue> array;
    std::string string;
    double number = 0.0;
};

class JsonParser
{
public:
    explicit JsonParser(std::string_view source)
        : text(source)
    {}

    JsonValue parse()
    {
        auto value = parseValue();
        skipWhitespace();
        if (!atEnd()) {
            throw std::runtime_error("Unexpected JSON content after root value");
        }
        return value;
    }

private:
    std::string_view text;
    size_t position = 0;

    bool atEnd() const
    {
        return position >= text.size();
    }

    char peek() const
    {
        if (atEnd()) {
            throw std::runtime_error("Unexpected end of JSON");
        }
        return text[position];
    }

    char consume()
    {
        const char value = peek();
        ++position;
        return value;
    }

    void expect(const char expected)
    {
        if (consume() != expected) {
            throw std::runtime_error("Unexpected JSON token");
        }
    }

    void skipWhitespace()
    {
        while (!atEnd()
               && std::isspace(static_cast<unsigned char>(text[position]))) {
            ++position;
        }
    }

    JsonValue parseValue()
    {
        skipWhitespace();
        const char next = peek();
        if (next == '{') {
            return parseObject();
        }
        if (next == '[') {
            return parseArray();
        }
        if (next == '"') {
            JsonValue value;
            value.type = JsonValue::Type::String;
            value.string = parseString();
            return value;
        }
        if (next == '-' || std::isdigit(static_cast<unsigned char>(next))) {
            return parseNumber();
        }
        throw std::runtime_error("Unsupported JSON value");
    }

    JsonValue parseObject()
    {
        JsonValue value;
        value.type = JsonValue::Type::Object;
        expect('{');
        skipWhitespace();

        if (!atEnd() && peek() == '}') {
            consume();
            return value;
        }

        while (true) {
            skipWhitespace();
            if (peek() != '"') {
                throw std::runtime_error("Expected JSON object key");
            }

            const auto key = parseString();
            skipWhitespace();
            expect(':');
            value.object.emplace(key, parseValue());
            skipWhitespace();

            const char separator = consume();
            if (separator == '}') {
                break;
            }
            if (separator != ',') {
                throw std::runtime_error("Expected JSON object separator");
            }
        }

        return value;
    }

    JsonValue parseArray()
    {
        JsonValue value;
        value.type = JsonValue::Type::Array;
        expect('[');
        skipWhitespace();

        if (!atEnd() && peek() == ']') {
            consume();
            return value;
        }

        while (true) {
            value.array.push_back(parseValue());
            skipWhitespace();

            const char separator = consume();
            if (separator == ']') {
                break;
            }
            if (separator != ',') {
                throw std::runtime_error("Expected JSON array separator");
            }
        }

        return value;
    }

    std::string parseString()
    {
        expect('"');
        std::string result;

        while (true) {
            if (atEnd()) {
                throw std::runtime_error("Unterminated JSON string");
            }

            const char value = consume();
            if (value == '"') {
                return result;
            }

            if (value == '\\') {
                if (atEnd()) {
                    throw std::runtime_error("Unterminated JSON escape");
                }
                const char escaped = consume();
                switch (escaped) {
                case '"':
                case '\\':
                case '/':
                    result.push_back(escaped);
                    break;
                case 'b':
                    result.push_back('\b');
                    break;
                case 'f':
                    result.push_back('\f');
                    break;
                case 'n':
                    result.push_back('\n');
                    break;
                case 'r':
                    result.push_back('\r');
                    break;
                case 't':
                    result.push_back('\t');
                    break;
                default:
                    throw std::runtime_error("Unsupported JSON string escape");
                }
            } else {
                result.push_back(value);
            }
        }
    }

    JsonValue parseNumber()
    {
        const size_t start = position;

        if (!atEnd() && peek() == '-') {
            consume();
        }
        while (!atEnd() && std::isdigit(static_cast<unsigned char>(peek()))) {
            consume();
        }
        if (!atEnd() && peek() == '.') {
            consume();
            while (!atEnd()
                   && std::isdigit(static_cast<unsigned char>(peek()))) {
                consume();
            }
        }
        if (!atEnd() && (peek() == 'e' || peek() == 'E')) {
            consume();
            if (!atEnd() && (peek() == '+' || peek() == '-')) {
                consume();
            }
            while (!atEnd()
                   && std::isdigit(static_cast<unsigned char>(peek()))) {
                consume();
            }
        }

        JsonValue value;
        value.type = JsonValue::Type::Number;
        value.number = std::stod(std::string(text.substr(start,
                                                         position - start)));
        return value;
    }
};

const JsonValue &requireObjectField(const JsonValue &value,
                                    const std::string &field)
{
    if (value.type != JsonValue::Type::Object) {
        throw std::runtime_error("Expected JSON object");
    }

    const auto found = value.object.find(field);
    if (found == value.object.end()) {
        throw std::runtime_error("Missing JSON field: " + field);
    }

    return found->second;
}

const JsonValue &requireObject(const JsonValue &value, const std::string &field)
{
    const auto &fieldValue = requireObjectField(value, field);
    if (fieldValue.type != JsonValue::Type::Object) {
        throw std::runtime_error("Expected JSON object field: " + field);
    }
    return fieldValue;
}

std::string requireString(const JsonValue &value, const std::string &field)
{
    const auto &fieldValue = requireObjectField(value, field);
    if (fieldValue.type != JsonValue::Type::String) {
        throw std::runtime_error("Expected JSON string field: " + field);
    }
    return fieldValue.string;
}

size_t requireSize(const JsonValue &value, const std::string &field)
{
    const auto &fieldValue = requireObjectField(value, field);
    if (fieldValue.type != JsonValue::Type::Number || fieldValue.number < 0.0
        || std::floor(fieldValue.number) != fieldValue.number) {
        throw std::runtime_error("Expected JSON unsigned integer field: "
                                 + field);
    }
    return static_cast<size_t>(fieldValue.number);
}

uint32_t requireSampleRate(const JsonValue &value)
{
    const auto sampleRate = requireSize(value, "sampleRate");
    return static_cast<uint32_t>(sampleRate);
}

std::vector<float> requireFloatArray(const JsonValue &value,
                                     const std::string &field)
{
    const auto &fieldValue = requireObjectField(value, field);
    if (fieldValue.type != JsonValue::Type::Array) {
        throw std::runtime_error("Expected JSON array field: " + field);
    }

    std::vector<float> values;
    values.reserve(fieldValue.array.size());
    for (const auto &item : fieldValue.array) {
        if (item.type != JsonValue::Type::Number) {
            throw std::runtime_error("Expected numeric JSON array item: "
                                     + field);
        }
        values.push_back(static_cast<float>(item.number));
    }
    return values;
}

void requireEquals(const std::string &actual,
                   const std::string &expected,
                   const std::string &field)
{
    if (actual != expected) {
        throw std::runtime_error("Unsupported JSON field value: " + field);
    }
}

FIRFilterModel toFIRFilterModel(const JsonValue &root)
{
    requireEquals(requireString(root, "format"),
                  "nn-training.fir-filter.v1",
                  "format");

    const auto &architecture = requireObject(root, "architecture");
    const auto &parameters = requireObject(root, "parameters");

    requireEquals(requireString(architecture, "type"), "Conv1D", "type");
    requireEquals(requireString(architecture, "activation"),
                  "identity",
                  "activation");

    FIRFilterModel model;
    model.sampleRate = requireSampleRate(root);
    model.weights.inputChannelCount = requireSize(architecture,
                                                  "inputChannels");
    model.weights.outputChannelCount = requireSize(architecture,
                                                   "outputChannels");
    model.weights.kernelLength = requireSize(architecture, "kernelSize");
    model.weights.stride = requireSize(architecture, "stride");
    model.weights.padding = requireSize(architecture, "padding");
    model.weights.weights = requireFloatArray(parameters, "weights");
    model.weights.biases = requireFloatArray(parameters, "biases");

    const size_t expectedWeightCount = model.weights.inputChannelCount
                                       * model.weights.outputChannelCount
                                       * model.weights.kernelLength;
    if (model.weights.weights.size() != expectedWeightCount) {
        throw std::runtime_error("FIR filter weight count does not match "
                                 "architecture");
    }
    if (model.weights.biases.size() != model.weights.outputChannelCount) {
        throw std::runtime_error("FIR filter bias count does not match "
                                 "architecture");
    }

    return model;
}

} // namespace

FIRFilterModel loadFIRFilterModelJson(const std::string_view json)
{
    return toFIRFilterModel(JsonParser(json).parse());
}

FIRFilterModel loadFIRFilterModelJsonFile(const std::string &path)
{
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Could not read FIR filter model: " + path);
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    return loadFIRFilterModelJson(buffer.str());
}

} // namespace audio::firFilter
