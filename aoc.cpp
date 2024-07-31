#include <iostream>
#include <iterator>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <valarray>

using namespace std;

int dayOne(vector<string> input, bool advanced) {

    int zeroVal = int('0');
    int nineVal = int('9');

    auto processLine = [zeroVal, nineVal](string line) {
        int first_digit = -1;
        int last_digit = -1;
        for (auto c : line) {
            int cVal = int(c);
            if (cVal >= zeroVal && cVal <= nineVal) {
                last_digit = cVal - zeroVal;
                first_digit = first_digit == -1 ? last_digit : first_digit;
            }
        }
        return 10 * first_digit + last_digit;
    };

    auto processLineAdvanced = [zeroVal, nineVal](string line) {
        int first_digit = -1;
        int last_digit = -1;
        int i = -1;
        while (i < (int)line.length()) {
            // manually move iter. var -> will manipulate with later
            i++;
            char c = line[i];
            int cVal = int(c);
            if (cVal > zeroVal && cVal <= nineVal) {
                last_digit = cVal - zeroVal;
                first_digit = first_digit == -1 ? last_digit : first_digit;
                continue;
            }
            int new_last_digit = -1;
            switch (c)
            {
            case 'o':
                if (!line.substr(i, 3).compare("one")) {
                    new_last_digit = 1;
                    i += 1;
                }
                break;
            case 't':
                if (!line.substr(i, 3).compare("two")) {
                    new_last_digit = 2;
                    i += 1;
                    break;
                }
                if (!line.substr(i, 5).compare("three")) {
                    new_last_digit = 3;
                    i += 3;
                    break;
                }
                break;
            case 'f':
                if (!line.substr(i, 4).compare("four")) {
                    new_last_digit = 4;
                    i += 3;
                    break;
                }
                if (!line.substr(i, 4).compare("five")) {
                    new_last_digit = 5;
                    i += 2;
                    break;
                }
                break;
            case 's':
                if (!line.substr(i, 3).compare("six")) {
                    new_last_digit = 6;
                    i += 2;
                    break;
                }
                if (!line.substr(i, 5).compare("seven")) {
                    new_last_digit = 7;
                    i += 3;
                    break;
                }
                break;
            case 'e':
                if (!line.substr(i, 5).compare("eight")) {
                    new_last_digit = 8;
                    i += 3;
                    break;
                }
                break;
            case 'n':
                if (!line.substr(i, 4).compare("nine")) {
                    new_last_digit = 9;
                    i += 2;
                    break;
                }
                break;            
            default:
                break;
            }
            if (new_last_digit > -1) {
                last_digit = new_last_digit;
                first_digit = first_digit == -1 ? last_digit : first_digit;
            }
        }
        return 10 * first_digit + last_digit;
    };


    long sum = 0;
    for (auto line : input) {
        long res = advanced ? processLineAdvanced(line) : processLine(line);
        //cout << res << endl;
        sum += res;
    }
    return sum;
}

int dayTwo(vector<string> input, bool advanced) {
    
    auto processLine = [advanced](string line, int id) {
        string cutLine = line.substr(line.find(": ")+2);
        cout << cutLine << endl;
        int red = 0; int green = 0; int blue = 0;
        int sum = 0;
        int i = 0;
        // state automaton
        do {
            char c = cutLine[i];
            switch (c)
            {
            case ' ':
                i++;
                break;
            case 'r':
                red = max(red, sum);
                sum = 0;
                i += 4;            
                break;
            case 'g':
                green = max(green, sum);
                sum = 0;
                i += 6;
                break;
            case 'b':
                blue = max(blue, sum);
                sum = 0;
                i += 5;
                break;
            default:
                sum = 10*sum + (int)c - (int)'0';
                i++;
                break;
            }            
        } while (i < cutLine.length());
        if (advanced) return red*green*blue;
        if (red <= 12 && green <= 13 && blue <= 14) {
            cout << id << endl;
            return id;
        }
        return 0;
    };
    
    long sum = 0;
    int id = 0;
    for (auto line : input) {
        sum += processLine(line, ++id);
    }
    
    return sum;
}

vector<int (*)(vector<string>, bool)> days = {
    dayOne, dayTwo
};

int main(int argc, char* argv[]) {
    
    int number = -1;
    bool advanced = false;

    // argument one: problem number
    if(argc > 1) number = stoi(argv[1]);

    // argument two: use advanced
    if(argc > 2) {
        if(strcmp(argv[2], "-a") == 0) advanced = true;
    }

    // read from file
    string fname = "inputs/day"+to_string(number)+".txt";

    ifstream ReadFile(fname);

    if (!ReadFile.is_open()) {
        throw;
    }

    vector<string> input_lines;
    string line;
    while (getline(ReadFile, line)) {
        input_lines.push_back(line);
    }

    ReadFile.close();

    int (*func)(vector<string>, bool);

    func = days[number-1];

    cout << func(input_lines, advanced) << endl;

    return EXIT_SUCCESS; 
}