
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <fstream>
#include <sstream>
#include <random> 
 
using namespace std;

// Define the structure of a decision tree node
struct Node {
    string label;
    map<string, Node*> children;
};
std::vector<std::vector<std::string>> read_csv(std::string filename) {
    std::vector<std::vector<std::string>> data;
    std::ifstream file(filename);
    std::string line;

    while (getline(file, line)) {
        std::vector<std::string> row;
        std::stringstream ss(line);
        std::string cell;

        while (getline(ss, cell, ',')) {
            row.push_back(cell);
        }

        data.push_back(row);
    }

    return data;
}

// Calculate the entropy of a dataset based on the class values
double entropy(const vector<string>& classes) {
    map<string, int> freq;
    for (const auto& c : classes) {
        freq[c]++;
    }
    double ent = 0.0;
    for (const auto& [_, count] : freq) {
        double p = static_cast<double>(count) / classes.size();
        ent -= p * log2(p);
    }
    return ent;
}

// Calculate the information gain of a dataset based on a given attribute
double information_gain(const vector<vector<string>>& data, int attr_idx) {
    vector<string> classes(data.size());
    for (int i = 0; i < data.size(); ++i) {
        classes[i] = data[i].back();
    }
    double parent_entropy = entropy(classes);
    map<string, vector<vector<string>>> splits;
    for (const auto& record : data) {
        splits[record[attr_idx]].push_back(record);
    }
    double child_entropy = 0.0;
    for (const auto& [_, subset] : splits) {
        double p = static_cast<double>(subset.size()) / data.size();
        for (const auto& r : subset) {
        child_entropy += p * entropy({r.back()});
}
    }
    return parent_entropy - child_entropy;
}

// Build a decision tree recursively using ID3
Node* build_tree(const vector<vector<string>>& data, const vector<int>& attr_indices) {
    Node* root = new Node();
    vector<string> classes(data.size());
    for (int i = 0; i < data.size(); ++i) {
        classes[i] = data[i].back();
    }
    // If all examples have the same class, return a leaf node with that class
    if (entropy(classes) == 0.0) {
        root->label = classes.front();
        return root;
    }
    // If there are no attributes left to split on, return a leaf node with the majority class
    if (attr_indices.empty()) {
        map<string, int> freq;
        for (const auto& c : classes) {
            freq[c]++;
        }
        root->label = max_element(freq.begin(), freq.end(), [](const auto& p1, const auto& p2) {
            return p1.second < p2.second;
        })->first;
        return root;
    }
    // Otherwise, select the attribute with the highest information gain and split on it
    int best_attr_idx = attr_indices.front();
    double best_gain = -1.0;
    for (const auto& i : attr_indices) {
        double gain = information_gain(data, i);
        if (gain > best_gain) {
            best_attr_idx = i;
            best_gain = gain;
        }
    }
    root->label = to_string(best_attr_idx);
    map<string, vector<vector<string>>> splits;
    for (const auto& record : data) {
        splits[record[best_attr_idx]].push_back(record);
    }
    vector<int> remaining_attr_indices;
    for (const auto& i : attr_indices) {
        if (i != best_attr_idx) {
            remaining_attr_indices.push_back(i);
        }
    }
    for (const auto& [attr_val, subset] : splits) {
        Node* child = build_tree(subset, remaining_attr_indices);
root->children[attr_val] = child;
}
return root;
}

// Classify a single record using a decision tree
string classify(const vector<string>& record, Node* root) {
if (root->children.empty()) {
return root->label;
}
string attr_val = record[stoi(root->label)];
if (root->children.find(attr_val) == root->children.end()) {
return root->label;
}
return classify(record, root->children[attr_val]);
}

// Generate a random subset of the dataset for bagging
vector<vector<string>> bagging(const vector<vector<string>>& data, int num_samples) {
vector<vector<string>> subset(num_samples);
random_device rd;
mt19937 gen(rd());
uniform_int_distribution<> dis(0, data.size() - 1);
for (auto& record : subset) {
record = data[dis(gen)];
}
return subset;
}

// Build a random forest using bagging and ID3
vector<Node*> build_forest(const vector<vector<string>>& data, int num_trees, int max_depth, int num_features) {
vector<Node*> forest(num_trees);
vector<int> attr_indices(data.front().size() - 1);
iota(attr_indices.begin(), attr_indices.end(), 0);
for (int i = 0; i < num_trees; ++i) {
vector<vector<string>> subset = bagging(data, data.size());
random_shuffle(attr_indices.begin(), attr_indices.end());
attr_indices.resize(num_features);
forest[i] = build_tree(subset, attr_indices);
}
return forest;
}

// Classify a record using a random forest
string classify_forest(const vector<string>& record, const vector<Node*>& forest) {
map<string, int> freq;
for (const auto& tree : forest) {
string prediction = classify(record, tree);
freq[prediction]++;
}
return max_element(freq.begin(), freq.end(), [](const auto& p1, const auto& p2) {
return p1.second < p2.second;
})->first;
}

// Test the accuracy of a classifier on a dataset
double test_classifier(const vector<vector<string>>& data, const vector<Node*>& forest) {
int num_correct = 0;
for (const auto& record : data) {
string prediction = classify_forest(record, forest);
//cout<<prediction<<record.back()<<endl;
if (prediction == record.back()) {
    
    num_correct++;
}

}
return static_cast<double>(num_correct) / data.size();
}
void free_tree(Node* root) {
    for (auto& child : root->children) {
        free_tree(child.second);
    }
    delete root;
}

 int main() {
// // Load the dataset
// /*vector<vector<string>> data = {{"Sunny", "Hot", "High", "Weak", "No"},
// {"Sunny", "Hot", "High", "Strong", "No"},
// {"Overcast", "Hot", "High", "Weak", "Yes"},
// {"Rain", "Mild", "High", "Weak", "Yes"},
// {"Rain", "Cool", "Normal", "Weak", "Yes"},
// {"Rain", "Cool", "Normal", "Strong", "No"},
// {"Overcast", "Cool", "Normal", "Strong", "Yes"},
// {"Sunny", "Mild", "High", "Weak", "No"},
// {"Sunny", "Cool", "Normal", "Weak", "Yes"},
// {"Rain", "Mild", "Normal", "Weak", "Yes"},
// {"Sunny", "Mild", "Normal", "Strong", "Yes"},
// {"Overcast", "Mild", "High", "Strong", "Yes"},
// {"Overcast","Mild", "Normal", "Weak", "Yes"},
// {"Rain", "Mild", "High", "Strong", "No"},
// {"Overcast", "Mild", "High", "Weak", "Yes"}};
// */
 vector<vector<string>> data=read_csv("titanic.csv");
     //for(int i=0;i<data[0].size();i++)cout<<data[0][i]<<" ";
//Split the dataset into training and test sets
vector<vector<string>> train_set, test_set;
random_device rd;
mt19937 gen(rd());
shuffle(data.begin(), data.end(), gen);
int num_train = data.size() * 0.7;
train_set.assign(data.begin(), data.begin() + num_train);
test_set.assign(data.begin() + num_train, data.end());

// Train a random forest
vector<Node*> forest = build_forest(train_set, 4, 3, 2);

// Test the accuracy of the random forest
double accuracy = test_classifier(test_set, forest);
cout << "Accuracy: " << accuracy << endl;

// Free the memory allocated for the decision trees
for (const auto& tree : forest) {
    free_tree(tree);
}

 return 0;
 }
 